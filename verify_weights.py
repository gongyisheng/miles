#!/usr/bin/env python3
"""
Verify torch_dist checkpoint weights against original HuggingFace weights.
This script compares key weights to identify potential corruption during conversion.

Usage: python verify_torch_dist_weights.py --hf-path /path/to/hf/model --torch-dist-path /path/to/torch_dist
"""

import argparse
import os
import torch
from safetensors import safe_open
from pathlib import Path


def load_hf_weights(hf_path):
    """Load weights from HuggingFace model."""
    weights = {}
    hf_path = Path(hf_path)

    # Try safetensors first
    safetensor_files = list(hf_path.glob("*.safetensors"))
    if safetensor_files:
        for sf_file in safetensor_files:
            with safe_open(sf_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    weights[key] = f.get_tensor(key)
    else:
        # Fall back to pytorch bin files
        bin_files = list(hf_path.glob("*.bin"))
        for bin_file in bin_files:
            state_dict = torch.load(bin_file, map_location="cpu")
            weights.update(state_dict)

    print(f"Loaded {len(weights)} weights from HuggingFace model")
    return weights


def load_torch_dist_weights(torch_dist_path):
    """Load weights from torch_dist checkpoint."""
    weights = {}
    torch_dist_path = Path(torch_dist_path)

    # Find the release checkpoint directory
    release_dir = torch_dist_path / "release"
    if not release_dir.exists():
        # Try iter_* directories
        iter_dirs = list(torch_dist_path.glob("iter_*"))
        if iter_dirs:
            release_dir = iter_dirs[0]
        else:
            raise FileNotFoundError(f"No checkpoint found in {torch_dist_path}")

    # Find model state files
    model_dir = release_dir / "model"
    if model_dir.exists():
        # torch_dist format with distcp
        for distcp_file in model_dir.glob("**/*.distcp"):
            state = torch.load(distcp_file, map_location="cpu")
            if isinstance(state, dict):
                weights.update(state)

        # Also check for .pt files
        for pt_file in model_dir.glob("**/*.pt"):
            state = torch.load(pt_file, map_location="cpu")
            if isinstance(state, dict):
                weights.update(state)
    else:
        # Try loading model_optim_rng.pt directly
        model_file = release_dir / "model_optim_rng.pt"
        if model_file.exists():
            state = torch.load(model_file, map_location="cpu")
            if "model" in state:
                weights = state["model"]
            else:
                weights = state

    print(f"Loaded {len(weights)} weights from torch_dist checkpoint")
    return weights


def get_megatron_to_hf_mapping():
    """Return mapping from Megatron weight names to HuggingFace names."""
    # Direct mappings
    direct = {
        "embedding.word_embeddings.weight": "model.embed_tokens.weight",
        "decoder.final_layernorm.weight": "model.norm.weight",
        "output_layer.weight": "lm_head.weight",
    }
    return direct


def compare_weights(hf_weights, megatron_weights, args):
    """Compare weights between HuggingFace and Megatron formats."""

    print("\n" + "="*60)
    print("Weight Comparison Report")
    print("="*60)

    # Check embedding weights
    hf_embed_key = "model.embed_tokens.weight"
    megatron_embed_keys = [k for k in megatron_weights.keys() if "embedding.word_embeddings.weight" in k]

    if hf_embed_key in hf_weights and megatron_embed_keys:
        hf_embed = hf_weights[hf_embed_key]
        megatron_embed = megatron_weights[megatron_embed_keys[0]]

        # Handle potential vocab padding
        vocab_size = min(hf_embed.shape[0], megatron_embed.shape[0])
        hf_embed_trimmed = hf_embed[:vocab_size]
        megatron_embed_trimmed = megatron_embed[:vocab_size]

        diff = (hf_embed_trimmed - megatron_embed_trimmed).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print(f"\n[Embedding] {hf_embed_key}")
        print(f"  HF shape: {hf_embed.shape}, Megatron shape: {megatron_embed.shape}")
        print(f"  Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
        print(f"  Status: {'OK' if max_diff < 1e-5 else 'MISMATCH!'}")

    # Check lm_head weights
    hf_lmhead_key = "lm_head.weight"
    megatron_lmhead_keys = [k for k in megatron_weights.keys() if "output_layer.weight" in k]

    if hf_lmhead_key in hf_weights and megatron_lmhead_keys:
        hf_lmhead = hf_weights[hf_lmhead_key]
        megatron_lmhead = megatron_weights[megatron_lmhead_keys[0]]

        vocab_size = min(hf_lmhead.shape[0], megatron_lmhead.shape[0])
        hf_lmhead_trimmed = hf_lmhead[:vocab_size]
        megatron_lmhead_trimmed = megatron_lmhead[:vocab_size]

        diff = (hf_lmhead_trimmed - megatron_lmhead_trimmed).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print(f"\n[LM Head] {hf_lmhead_key}")
        print(f"  HF shape: {hf_lmhead.shape}, Megatron shape: {megatron_lmhead.shape}")
        print(f"  Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
        print(f"  Status: {'OK' if max_diff < 1e-5 else 'MISMATCH!'}")
    elif hf_lmhead_key not in hf_weights:
        print(f"\n[LM Head] Not in HF weights (tie_word_embeddings=True)")
        # Check if Megatron has separate output_layer
        if megatron_lmhead_keys:
            print(f"  WARNING: Megatron has output_layer.weight but HF uses tied embeddings!")
            print(f"  This could cause issues if the conversion didn't handle tied embeddings correctly.")

    # Check a few layer weights
    print("\n[Layer 0 QKV Weight Check]")

    # Find QKV weights in Megatron format
    qkv_keys = [k for k in megatron_weights.keys() if "layers.0" in k and "linear_qkv.weight" in k]
    if qkv_keys:
        megatron_qkv = megatron_weights[qkv_keys[0]]
        print(f"  Megatron QKV shape: {megatron_qkv.shape}")

        # Load separate Q, K, V from HF
        q_key = "model.layers.0.self_attn.q_proj.weight"
        k_key = "model.layers.0.self_attn.k_proj.weight"
        v_key = "model.layers.0.self_attn.v_proj.weight"

        if all(k in hf_weights for k in [q_key, k_key, v_key]):
            hf_q = hf_weights[q_key]
            hf_k = hf_weights[k_key]
            hf_v = hf_weights[v_key]

            print(f"  HF Q shape: {hf_q.shape}, K shape: {hf_k.shape}, V shape: {hf_v.shape}")

            # For GQA: Q has more heads than K/V
            # Megatron QKV is packed as [num_query_groups, (q_per_group + 1 + 1), head_dim, hidden]
            num_q_heads = hf_q.shape[0] // (hf_q.shape[1] // (hf_weights.get("model.embed_tokens.weight", hf_q).shape[1] if "model.embed_tokens.weight" in hf_weights else hf_q.shape[1]))

            # Try to reconstruct QKV from HF weights
            expected_qkv_size = hf_q.shape[0] + hf_k.shape[0] + hf_v.shape[0]
            print(f"  Expected combined QKV size: {expected_qkv_size} (Q:{hf_q.shape[0]} + K:{hf_k.shape[0]} + V:{hf_v.shape[0]})")
            print(f"  Megatron QKV first dim: {megatron_qkv.shape[0]}")

            if megatron_qkv.shape[0] != expected_qkv_size:
                print(f"  WARNING: QKV dimension mismatch! This could indicate incorrect GQA handling.")

    # Check MLP weights
    print("\n[Layer 0 MLP Weight Check]")
    gate_up_keys = [k for k in megatron_weights.keys() if "layers.0" in k and "linear_fc1.weight" in k]
    if gate_up_keys:
        megatron_fc1 = megatron_weights[gate_up_keys[0]]
        print(f"  Megatron FC1 (gate+up) shape: {megatron_fc1.shape}")

        hf_gate = hf_weights.get("model.layers.0.mlp.gate_proj.weight")
        hf_up = hf_weights.get("model.layers.0.mlp.up_proj.weight")

        if hf_gate is not None and hf_up is not None:
            print(f"  HF gate shape: {hf_gate.shape}, up shape: {hf_up.shape}")

            # Reconstruct and compare
            hf_fc1 = torch.cat([hf_gate, hf_up], dim=0)
            diff = (hf_fc1 - megatron_fc1).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()

            print(f"  Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}")
            print(f"  Status: {'OK' if max_diff < 1e-5 else 'MISMATCH!'}")

    # Print all available keys for debugging
    print("\n" + "="*60)
    print("Available Megatron weight keys (first 20):")
    print("="*60)
    for i, key in enumerate(sorted(megatron_weights.keys())[:20]):
        print(f"  {key}: {megatron_weights[key].shape}")

    print("\n" + "="*60)
    print("Available HF weight keys (first 20):")
    print("="*60)
    for i, key in enumerate(sorted(hf_weights.keys())[:20]):
        print(f"  {key}: {hf_weights[key].shape}")


def main():
    parser = argparse.ArgumentParser(description="Verify torch_dist checkpoint against HF weights")
    parser.add_argument("--hf-path", type=str, required=True, help="Path to HuggingFace model")
    parser.add_argument("--torch-dist-path", type=str, required=True, help="Path to torch_dist checkpoint")
    args = parser.parse_args()

    print(f"Loading HuggingFace weights from: {args.hf_path}")
    hf_weights = load_hf_weights(args.hf_path)

    print(f"Loading torch_dist weights from: {args.torch_dist_path}")
    megatron_weights = load_torch_dist_weights(args.torch_dist_path)

    compare_weights(hf_weights, megatron_weights, args)


if __name__ == "__main__":
    main()