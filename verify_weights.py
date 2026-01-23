#!/usr/bin/env python3
"""
Load Megatron DCP checkpoint and compare with HuggingFace weights.

Usage: python load_dcp_checkpoint.py /path/to/torch_dist [/path/to/hf_model]
"""

import sys
import json
import torch
from pathlib import Path


def load_dcp_checkpoint(ckpt_path):
    """Load all tensors from a DCP checkpoint."""
    ckpt_path = Path(ckpt_path)

    # Find the release directory
    release_dir = ckpt_path / "release"
    if not release_dir.exists():
        release_dir = ckpt_path

    print(f"Loading from: {release_dir}")
    print(f"Directory contents:")
    for item in sorted(release_dir.iterdir()):
        size = item.stat().st_size if item.is_file() else 0
        print(f"  {item.name} ({size:,} bytes)")

    weights = {}

    # Load metadata.json to understand structure
    metadata_file = release_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        print(f"\nmetadata.json keys: {list(metadata.keys())}")

    # Load common.pt to see what's there
    common_file = release_dir / "common.pt"
    if common_file.exists():
        print(f"\nLoading common.pt...")
        common_data = torch.load(common_file, map_location="cpu", weights_only=False)
        print(f"common.pt type: {type(common_data)}")
        if isinstance(common_data, dict):
            print(f"common.pt keys: {list(common_data.keys())[:20]}")

    # Load __0_0.distcp and __0_1.distcp
    # These contain the actual model weights
    distcp_files = sorted(release_dir.glob("__*.distcp"))
    print(f"\nFound {len(distcp_files)} .distcp files")

    for distcp_file in distcp_files:
        print(f"\nLoading {distcp_file.name}...")
        try:
            data = torch.load(distcp_file, map_location="cpu", weights_only=False)
            print(f"  Type: {type(data)}")

            if isinstance(data, dict):
                print(f"  Keys ({len(data)}):")
                for key in sorted(data.keys())[:30]:
                    val = data[key]
                    if isinstance(val, torch.Tensor):
                        print(f"    {key}: {val.shape} {val.dtype}")
                    else:
                        print(f"    {key}: {type(val)}")
                if len(data) > 30:
                    print(f"    ... and {len(data) - 30} more")
                weights.update(data)

            elif isinstance(data, torch.Tensor):
                print(f"  Tensor: {data.shape} {data.dtype}")
                # Use filename as key
                weights[distcp_file.stem] = data

            else:
                print(f"  Unknown format: {type(data)}")
                # Try to iterate if possible
                if hasattr(data, '__iter__'):
                    for i, item in enumerate(data):
                        if i < 5:
                            print(f"    [{i}]: {type(item)}")

        except Exception as e:
            print(f"  Error loading: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Total loaded: {len(weights)} weight entries")
    print(f"{'='*60}")

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

    if not hf_weights:
        # Try .bin files
        for bin_file in hf_path.glob("*.bin"):
            state = torch.load(bin_file, map_location="cpu")
            hf_weights.update(state)

    print(f"\nLoaded {len(hf_weights)} HF weights")
    print(f"Loaded {len(weights)} Megatron weights\n")

    if len(weights) == 0:
        print("No Megatron weights loaded - cannot compare")
        return

    # Show some HF keys for reference
    print("HF weight keys (first 10):")
    for key in sorted(hf_weights.keys())[:10]:
        print(f"  {key}: {hf_weights[key].shape}")

    print("\nMegatron weight keys (first 10):")
    for key in sorted(weights.keys())[:10]:
        val = weights[key]
        if isinstance(val, torch.Tensor):
            print(f"  {key}: {val.shape}")
        else:
            print(f"  {key}: {type(val)}")

    # Try to find and compare embedding
    print("\n" + "="*60)
    print("Weight Comparison")
    print("="*60)

    # Find embedding in megatron weights
    embed_keys = [k for k in weights.keys() if "embed" in k.lower() or "word_embedding" in k.lower()]
    if embed_keys:
        print(f"\nEmbedding keys found: {embed_keys}")
        for ek in embed_keys:
            if isinstance(weights[ek], torch.Tensor):
                meg_embed = weights[ek]
                hf_embed = hf_weights.get("model.embed_tokens.weight")
                if hf_embed is not None:
                    vocab_size = min(hf_embed.shape[0], meg_embed.shape[0])
                    diff = (hf_embed[:vocab_size] - meg_embed[:vocab_size]).abs()
                    print(f"  {ek}:")
                    print(f"    HF shape: {hf_embed.shape}, Megatron shape: {meg_embed.shape}")
                    print(f"    Max diff: {diff.max().item():.6e}, Mean diff: {diff.mean().item():.6e}")
                    print(f"    Status: {'OK' if diff.max().item() < 1e-5 else 'MISMATCH!'}")

    # Find QKV in megatron weights
    qkv_keys = [k for k in weights.keys() if "qkv" in k.lower() or "q_proj" in k.lower()]
    if qkv_keys:
        print(f"\nQKV keys found: {qkv_keys[:5]}")

    # Find layer norm
    ln_keys = [k for k in weights.keys() if "layernorm" in k.lower() or "layer_norm" in k.lower() or "norm" in k.lower()]
    if ln_keys:
        print(f"\nLayerNorm keys found: {ln_keys[:5]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python load_dcp_checkpoint.py <torch_dist_path> [hf_path]")
        sys.exit(1)

    ckpt_path = sys.argv[1]
    weights = load_dcp_checkpoint(ckpt_path)

    if len(sys.argv) >= 3:
        hf_path = sys.argv[2]
        compare_with_hf(weights, hf_path)