#!/usr/bin/env python3
"""
Load DCP (Distributed Checkpoint) format and print tensor info.
DCP stores each tensor in a directory with __0_0.distcp files.

Usage: python load_dcp_checkpoint.py /path/to/torch_dist
"""

import sys
import pickle
import torch
from pathlib import Path


def load_dcp_checkpoint(ckpt_path):
    """Load all tensors from a DCP checkpoint."""
    ckpt_path = Path(ckpt_path)

    # Find the model directory
    model_dir = ckpt_path / "release" / "model"
    if not model_dir.exists():
        model_dir = ckpt_path / "release"
    if not model_dir.exists():
        model_dir = ckpt_path

    print(f"Loading from: {model_dir}")

    # Check for .metadata file
    metadata_file = model_dir / ".metadata"
    if not metadata_file.exists():
        print(f"ERROR: No .metadata file found in {model_dir}")
        print("Directory contents:")
        for item in sorted(model_dir.iterdir())[:20]:
            print(f"  {item.name}")
        return {}

    # Load metadata
    with open(metadata_file, "rb") as f:
        metadata = pickle.load(f)

    print(f"Metadata type: {type(metadata).__name__}")

    weights = {}

    # Get the state dict metadata
    if hasattr(metadata, "state_dict_metadata"):
        sd_meta = metadata.state_dict_metadata
        print(f"Found {len(sd_meta)} tensor entries in metadata\n")

        for key in sorted(sd_meta.keys()):
            tensor_meta = sd_meta[key]

            # Find the .distcp file for this tensor
            # DCP stores tensors in directories named after the key
            # with files like __0_0.distcp for rank 0
            tensor_dir = model_dir / key
            distcp_file = tensor_dir / "__0_0.distcp"

            if not distcp_file.exists():
                # Some tensors might be stored differently
                # Try direct file
                distcp_file = model_dir / f"{key}/__0_0.distcp"

            if distcp_file.exists():
                try:
                    tensor = torch.load(distcp_file, map_location="cpu", weights_only=False)
                    weights[key] = tensor
                    print(f"  {key}: {tensor.shape} {tensor.dtype}")
                except Exception as e:
                    print(f"  {key}: ERROR loading - {e}")
            else:
                # Check if it's stored as a single chunk
                # or find any distcp file in the tensor dir
                if tensor_dir.exists():
                    distcp_files = list(tensor_dir.glob("*.distcp"))
                    if distcp_files:
                        try:
                            tensor = torch.load(distcp_files[0], map_location="cpu", weights_only=False)
                            weights[key] = tensor
                            print(f"  {key}: {tensor.shape} {tensor.dtype}")
                        except Exception as e:
                            print(f"  {key}: ERROR loading - {e}")
                    else:
                        print(f"  {key}: No .distcp file found in {tensor_dir}")
                else:
                    print(f"  {key}: Directory not found")

    return weights


def compare_with_hf(weights, hf_path):
    """Compare loaded weights with HuggingFace model."""
    from safetensors import safe_open

    hf_path = Path(hf_path)
    hf_weights = {}

    # Load HF weights
    for sf_file in hf_path.glob("*.safetensors"):
        with safe_open(sf_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                hf_weights[key] = f.get_tensor(key)

    print(f"\nLoaded {len(hf_weights)} HF weights")
    print(f"Loaded {len(weights)} Megatron weights\n")

    # Compare embedding
    if "model.embed_tokens.weight" in hf_weights:
        hf_embed = hf_weights["model.embed_tokens.weight"]

        # Find megatron embedding
        meg_embed_key = [k for k in weights.keys() if "embedding.word_embeddings.weight" in k]
        if meg_embed_key:
            meg_embed = weights[meg_embed_key[0]]
            vocab_size = min(hf_embed.shape[0], meg_embed.shape[0])

            diff = (hf_embed[:vocab_size] - meg_embed[:vocab_size]).abs()
            print(f"Embedding comparison:")
            print(f"  HF shape: {hf_embed.shape}, Megatron shape: {meg_embed.shape}")
            print(f"  Max diff: {diff.max().item():.6e}, Mean diff: {diff.mean().item():.6e}")
            if diff.max().item() < 1e-5:
                print(f"  Status: OK")
            else:
                print(f"  Status: MISMATCH!")

    # Compare first layer QKV
    q_key = "model.layers.0.self_attn.q_proj.weight"
    if q_key in hf_weights:
        hf_q = hf_weights[q_key]
        hf_k = hf_weights["model.layers.0.self_attn.k_proj.weight"]
        hf_v = hf_weights["model.layers.0.self_attn.v_proj.weight"]

        # Find megatron QKV (it's packed together)
        qkv_key = [k for k in weights.keys() if "layers.0" in k and "linear_qkv.weight" in k]
        if qkv_key:
            meg_qkv = weights[qkv_key[0]]
            print(f"\nQKV comparison (Layer 0):")
            print(f"  HF Q: {hf_q.shape}, K: {hf_k.shape}, V: {hf_v.shape}")
            print(f"  Megatron QKV: {meg_qkv.shape}")

            # For GQA, expected QKV dim = Q + K + V
            expected_dim = hf_q.shape[0] + hf_k.shape[0] + hf_v.shape[0]
            print(f"  Expected combined: {expected_dim}, Actual: {meg_qkv.shape[0]}")

            if meg_qkv.shape[0] == expected_dim:
                # Try simple concatenation (Q, K, V order)
                hf_qkv_simple = torch.cat([hf_q, hf_k, hf_v], dim=0)
                diff_simple = (hf_qkv_simple - meg_qkv).abs()

                print(f"  Testing [Q,K,V] order: max_diff={diff_simple.max().item():.6e}")

                if diff_simple.max().item() > 1e-5:
                    # Try [Q,V,K] order (sometimes used)
                    hf_qkv_alt = torch.cat([hf_q, hf_v, hf_k], dim=0)
                    diff_alt = (hf_qkv_alt - meg_qkv).abs()
                    print(f"  Testing [Q,V,K] order: max_diff={diff_alt.max().item():.6e}")

                    # Check if it's interleaved
                    # Megatron GQA format: [num_query_groups, q_per_group+1+1, head_dim, hidden]
                    print(f"\n  Checking interleaved GQA format...")
                    num_query_groups = hf_k.shape[0] // 64  # Assuming head_dim=64
                    head_dim = 64
                    q_per_group = hf_q.shape[0] // num_query_groups // head_dim

                    print(f"  num_query_groups={num_query_groups}, q_per_group={q_per_group}, head_dim={head_dim}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python load_dcp_checkpoint.py <torch_dist_path> [hf_path]")
        sys.exit(1)

    ckpt_path = sys.argv[1]
    weights = load_dcp_checkpoint(ckpt_path)

    if len(sys.argv) >= 3:
        hf_path = sys.argv[2]
        compare_with_hf(weights, hf_path)