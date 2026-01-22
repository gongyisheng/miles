"""
Minimal reproduction of the update_weight flow from Megatron to SGLang.

This script demonstrates the transformation pipeline:
1. Megatron model params (sharded across TP ranks)
2. all_gather_param: gather sharded params to full tensor
3. convert_to_hf: convert Megatron param names to HuggingFace names
4. Send to SGLang via NCCL broadcast

Run: python scripts/debug_update_weight_flow.py
"""

import torch


# =============================================================================
# Step 1: Simulate Megatron model parameters
# =============================================================================
# In real code, these come from: named_params_and_buffers(args, model)
# Located at: miles/backends/megatron_utils/update_weight/common.py:116

def simulate_megatron_params():
    """
    Simulate what named_params_and_buffers() yields.

    Real Megatron param names look like:
    - module.module.decoder.layers.{layer_idx}.{component}
    - module.module.embedding.word_embeddings.weight
    """
    # Simulated params for a small model (e.g., 2 layers, hidden=512, heads=8)
    hidden_size = 512
    num_heads = 8
    head_dim = hidden_size // num_heads
    intermediate_size = 2048
    vocab_size = 32000

    params = [
        # Embedding
        ("module.module.embedding.word_embeddings.weight", torch.randn(vocab_size, hidden_size)),

        # Layer 0 - attention
        ("module.module.decoder.layers.0.self_attention.linear_qkv.weight",
         torch.randn(3 * hidden_size, hidden_size)),  # fused QKV
        ("module.module.decoder.layers.0.self_attention.linear_qkv.bias",
         torch.randn(3 * hidden_size)),
        ("module.module.decoder.layers.0.self_attention.linear_proj.weight",
         torch.randn(hidden_size, hidden_size)),

        # Layer 0 - MLP (with GLU, so fc1 is 2x intermediate)
        ("module.module.decoder.layers.0.mlp.linear_fc1.weight",
         torch.randn(2 * intermediate_size, hidden_size)),  # gate + up fused
        ("module.module.decoder.layers.0.mlp.linear_fc2.weight",
         torch.randn(hidden_size, intermediate_size)),

        # Layer 0 - norms
        ("module.module.decoder.layers.0.self_attention.linear_qkv.layer_norm_weight",
         torch.randn(hidden_size)),
        ("module.module.decoder.layers.0.mlp.linear_fc1.layer_norm_weight",
         torch.randn(hidden_size)),

        # Final norm
        ("module.module.decoder.final_layernorm.weight", torch.randn(hidden_size)),

        # Output (LM head)
        ("module.module.output_layer.weight", torch.randn(vocab_size, hidden_size)),
    ]

    return params


# =============================================================================
# Step 2: Simulate all_gather_param (TP gathering)
# =============================================================================
# Real code at: miles/backends/megatron_utils/update_weight/common.py:15

def simulate_all_gather_param(name: str, param: torch.Tensor, tp_size: int = 1) -> torch.Tensor:
    """
    In real code, this gathers TP-sharded params across ranks.

    For TP=1 (no tensor parallelism), params are returned as-is.
    For TP>1, params are gathered and concatenated.

    Special handling:
    - linear_fc1.weight: GLU rechunking (gate/up split)
    - linear_fc2.weight: dimension fix for grouped MoE
    - .experts. params: use expert-TP group instead of regular-TP
    """
    print(f"[all_gather_param] input: {name}, shape: {param.shape}")

    if tp_size == 1:
        # No gathering needed
        print(f"[all_gather_param] output (TP=1, no gather): {name}, shape: {param.shape}")
        return param

    # Simulate gathering (in real code, this uses dist.all_gather)
    # For demo, just show what would happen
    if "linear_qkv" in name and "weight" in name:
        # QKV is sharded on dim 0
        gathered_shape = (param.shape[0] * tp_size, *param.shape[1:])
    elif "linear_fc1" in name and "weight" in name:
        # FC1 (gate+up) is sharded on dim 0, needs GLU rechunking
        gathered_shape = (param.shape[0] * tp_size, *param.shape[1:])
    elif "linear_fc2" in name and "weight" in name:
        # FC2 is sharded on dim 1
        gathered_shape = (param.shape[0], param.shape[1] * tp_size)
    elif "linear_proj" in name and "weight" in name:
        # Projection is sharded on dim 1
        gathered_shape = (param.shape[0], param.shape[1] * tp_size)
    else:
        gathered_shape = param.shape

    gathered = torch.randn(gathered_shape)
    print(f"[all_gather_param] output (TP={tp_size}, gathered): {name}, shape: {gathered.shape}")
    return gathered


# =============================================================================
# Step 3: Simulate convert_to_hf (Megatron -> HuggingFace name conversion)
# =============================================================================
# Real code at: miles/backends/megatron_utils/megatron_to_hf/__init__.py:20

def simulate_convert_to_hf(name: str, param: torch.Tensor, model_type: str = "llama"):
    """
    Convert Megatron param names to HuggingFace format.

    Different model types have different converters:
    - llama: convert_llama_to_hf
    - qwen2/qwen3: convert_qwen2_to_hf
    - deepseekv3: convert_deepseekv3_to_hf
    - glm4: convert_glm4_to_hf
    etc.

    Key transformations:
    - module.module.decoder.layers.X -> model.layers.X
    - linear_qkv -> split into q_proj, k_proj, v_proj
    - linear_fc1 -> split into gate_proj, up_proj (for GLU)
    - linear_fc2 -> down_proj
    - embedding.word_embeddings -> model.embed_tokens
    """
    print(f"[convert_to_hf] input: {name}, shape: {param.shape}")

    converted = []

    # Embedding
    if "embedding.word_embeddings.weight" in name:
        converted.append(("model.embed_tokens.weight", param))

    # Output layer (LM head)
    elif "output_layer.weight" in name:
        converted.append(("lm_head.weight", param))

    # Final layernorm
    elif "final_layernorm.weight" in name:
        converted.append(("model.norm.weight", param))

    # Decoder layers
    elif "decoder.layers" in name:
        # Extract layer index
        import re
        match = re.search(r"decoder\.layers\.(\d+)\.", name)
        if match:
            layer_idx = match.group(1)

            # QKV projection - split into separate Q, K, V
            if "self_attention.linear_qkv.weight" in name:
                hidden_size = param.shape[1]
                # Assuming equal Q, K, V sizes for simplicity
                q_size = k_size = v_size = param.shape[0] // 3
                q, k, v = param.split([q_size, k_size, v_size], dim=0)
                converted.append((f"model.layers.{layer_idx}.self_attn.q_proj.weight", q))
                converted.append((f"model.layers.{layer_idx}.self_attn.k_proj.weight", k))
                converted.append((f"model.layers.{layer_idx}.self_attn.v_proj.weight", v))

            elif "self_attention.linear_qkv.bias" in name:
                q_size = k_size = v_size = param.shape[0] // 3
                q, k, v = param.split([q_size, k_size, v_size], dim=0)
                converted.append((f"model.layers.{layer_idx}.self_attn.q_proj.bias", q))
                converted.append((f"model.layers.{layer_idx}.self_attn.k_proj.bias", k))
                converted.append((f"model.layers.{layer_idx}.self_attn.v_proj.bias", v))

            # Output projection
            elif "self_attention.linear_proj.weight" in name:
                converted.append((f"model.layers.{layer_idx}.self_attn.o_proj.weight", param))

            # MLP - FC1 is gate+up fused (GLU)
            elif "mlp.linear_fc1.weight" in name:
                gate, up = param.chunk(2, dim=0)
                converted.append((f"model.layers.{layer_idx}.mlp.gate_proj.weight", gate))
                converted.append((f"model.layers.{layer_idx}.mlp.up_proj.weight", up))

            # MLP - FC2 is down projection
            elif "mlp.linear_fc2.weight" in name:
                converted.append((f"model.layers.{layer_idx}.mlp.down_proj.weight", param))

            # Input layernorm (before attention)
            elif "self_attention.linear_qkv.layer_norm_weight" in name:
                converted.append((f"model.layers.{layer_idx}.input_layernorm.weight", param))

            # Post-attention layernorm (before MLP)
            elif "mlp.linear_fc1.layer_norm_weight" in name:
                converted.append((f"model.layers.{layer_idx}.post_attention_layernorm.weight", param))

    else:
        # Unknown param, pass through with warning
        print(f"[convert_to_hf] WARNING: unknown param {name}, passing through")
        converted.append((name, param))

    for hf_name, hf_param in converted:
        print(f"[convert_to_hf] output: {hf_name}, shape: {hf_param.shape}")

    return converted


# =============================================================================
# Step 4: Simulate sending to SGLang
# =============================================================================
# Real code at: miles/backends/megatron_utils/update_weight/update_weight_from_distributed.py:285

def simulate_send_to_sglang(converted_params: list):
    """
    In real code, this:
    1. Sends metadata (names, dtypes, shapes) via Ray remote call
    2. Broadcasts tensor data via NCCL from training rank 0 to SGLang engines

    SGLang receives via: engine.update_weights_from_distributed()
    Which calls the /update_weights_from_distributed HTTP endpoint
    """
    print("\n" + "="*60)
    print("[send_to_sglang] Sending weights to SGLang engine:")
    print("="*60)

    for hf_name, hf_param in converted_params:
        print(f"  {hf_name}: shape={hf_param.shape}, dtype={hf_param.dtype}")

    print("\nIn real code, this would:")
    print("  1. ray.get([engine.pause_generation.remote() for engine in engines])")
    print("  2. ray.get([engine.flush_cache.remote() for engine in engines])")
    print("  3. For each param bucket:")
    print("     - dist.broadcast(param.data, src=0, group=nccl_group)")
    print("  4. ray.get([engine.continue_generation.remote() for engine in engines])")


# =============================================================================
# Main: Run the full flow
# =============================================================================

def main():
    print("="*60)
    print("UPDATE_WEIGHT FLOW DEMONSTRATION")
    print("="*60)
    print("""
This demonstrates the weight update flow from Megatron training to SGLang inference:

1. named_params_and_buffers() - Iterate Megatron model params
   Location: miles/backends/megatron_utils/update_weight/common.py:116

2. all_gather_param() - Gather TP-sharded params to full tensors
   Location: miles/backends/megatron_utils/update_weight/common.py:15

3. convert_to_hf() - Convert Megatron names to HuggingFace format
   Location: miles/backends/megatron_utils/megatron_to_hf/__init__.py:20

4. update_weights_from_distributed() - NCCL broadcast to SGLang
   Location: miles/backends/megatron_utils/update_weight/update_weight_from_distributed.py:285
""")

    # Simulate TP=1 for simplicity
    tp_size = 1

    # Get simulated Megatron params
    megatron_params = simulate_megatron_params()

    all_converted = []

    print("\n" + "="*60)
    print("PROCESSING EACH PARAMETER:")
    print("="*60)

    for name, param in megatron_params:
        print(f"\n--- Processing: {name} ---")

        # Step 2: Gather across TP (no-op for TP=1)
        gathered_param = simulate_all_gather_param(name, param, tp_size)

        # Step 3: Convert to HuggingFace format
        converted = simulate_convert_to_hf(name, gathered_param)
        all_converted.extend(converted)

    # Step 4: Send to SGLang
    simulate_send_to_sglang(all_converted)

    print("\n" + "="*60)
    print("SUMMARY: Megatron -> HuggingFace name mapping")
    print("="*60)
    mapping = [
        ("module.module.embedding.word_embeddings.weight", "model.embed_tokens.weight"),
        ("module.module.decoder.layers.X.self_attention.linear_qkv.weight", "model.layers.X.self_attn.{q,k,v}_proj.weight"),
        ("module.module.decoder.layers.X.self_attention.linear_proj.weight", "model.layers.X.self_attn.o_proj.weight"),
        ("module.module.decoder.layers.X.mlp.linear_fc1.weight", "model.layers.X.mlp.{gate,up}_proj.weight"),
        ("module.module.decoder.layers.X.mlp.linear_fc2.weight", "model.layers.X.mlp.down_proj.weight"),
        ("module.module.decoder.layers.X.self_attention.linear_qkv.layer_norm_weight", "model.layers.X.input_layernorm.weight"),
        ("module.module.decoder.layers.X.mlp.linear_fc1.layer_norm_weight", "model.layers.X.post_attention_layernorm.weight"),
        ("module.module.decoder.final_layernorm.weight", "model.norm.weight"),
        ("module.module.output_layer.weight", "lm_head.weight"),
    ]
    for megatron_name, hf_name in mapping:
        print(f"  {megatron_name}")
        print(f"    -> {hf_name}")
        print()


if __name__ == "__main__":
    main()
