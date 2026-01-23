import os
import sys
import torch
import torch.distributed as dist

# Set environment variables before importing megatron
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
os.environ.setdefault("NVTE_FLASH_ATTN", "1")

def initialize_megatron():
    """Initialize Megatron environment with minimal args."""
    from megatron.training.initialize import initialize_megatron
    from megatron.training.arguments import parse_args

    # Minimal args for Qwen2.5-0.5B
    sys.argv = [
        sys.argv[0],
        "--num-layers", "24",
        "--hidden-size", "896",
        "--ffn-hidden-size", "4864",
        "--num-attention-heads", "14",
        "--seq-length", "2048",
        "--max-position-embeddings", "32768",
        "--micro-batch-size", "1",
        "--global-batch-size", "1",
        "--vocab-size", "151936",
        "--tokenizer-type", "HuggingFaceTokenizer",
        "--tokenizer-model", "/root/Qwen2.5-0.5B-Instruct/",  # Change this path
        "--load", "/root/Qwen2.5-0.5B-Instruct_torch_dist/",  # Change this path
        "--bf16",
        "--use-rotary-position-embeddings",
        "--disable-bias-linear",
        "--add-qkv-bias",
        "--normalization", "RMSNorm",
        "--norm-epsilon", "1e-6",
        "--rotary-base", "1000000",
        "--group-query-attention",
        "--num-query-groups", "2",
        "--swiglu",
        "--tensor-model-parallel-size", "1",
        "--pipeline-model-parallel-size", "1",
        "--no-masked-softmax-fusion",
        "--attention-softmax-in-fp32",
        "--no-load-optim",
        "--no-load-rng",
        "--untie-embeddings-and-output-weights",  # Try both with and without this
    ]

    initialize_megatron()


def load_model():
    """Load model and checkpoint."""
    from megatron.core.enums import ModelType
    from megatron.training.training import get_model
    from megatron.training.global_vars import get_args

    # Import after megatron init
    from miles.backends.megatron_utils.model_provider import get_model_provider_func
    from miles.backends.megatron_utils.checkpoint import load_checkpoint

    args = get_args()

    # Get model
    model = get_model(get_model_provider_func(args, "actor"), ModelType.encoder_or_decoder)

    # Load checkpoint
    iteration, _ = load_checkpoint(
        model, None, None,
        checkpointing_context={},
        skip_load_to_model_and_opt=False
    )

    print(f"Loaded checkpoint at iteration {iteration}")

    # Set to eval mode
    for m in model:
        m.eval()

    return model


def generate_text(model, prompt: str, max_new_tokens: int = 100):
    """Simple greedy generation."""
    from megatron.training.global_vars import get_tokenizer, get_args
    from megatron.core import mpu

    args = get_args()
    tokenizer = get_tokenizer()

    # Tokenize input
    input_ids = tokenizer.tokenize(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device="cuda")

    print(f"Input tokens: {input_ids[:20]}...")
    print(f"Input text: {prompt[:200]}...")

    generated_ids = list(input_ids)

    with torch.no_grad():
        for i in range(max_new_tokens):
            # Create position ids
            seq_len = len(generated_ids)
            position_ids = torch.arange(seq_len, dtype=torch.long, device="cuda").unsqueeze(0)
            input_tensor = torch.tensor([generated_ids], dtype=torch.long, device="cuda")

            # Forward pass
            output = model[0](
                input_ids=input_tensor,
                position_ids=position_ids,
                attention_mask=None,
                labels=None,
            )

            # Get logits for last token
            if hasattr(output, 'logits'):
                logits = output.logits
            else:
                logits = output

            last_logits = logits[0, -1, :]

            # Greedy: pick highest probability token
            next_token = torch.argmax(last_logits).item()

            generated_ids.append(next_token)

            # Check for EOS
            if next_token == tokenizer.eod:
                break

            if i < 10 or i % 20 == 0:
                print(f"Step {i}: token={next_token}, logit={last_logits[next_token]:.4f}")

    # Decode
    output_text = tokenizer.detokenize(generated_ids)
    return output_text


def main():
    print("Initializing Megatron...")
    initialize_megatron()

    print("Loading model...")
    model = load_model()

    # Test prompt (Qwen chat format)
    prompt = """<|im_start|>system
You are a helpful assistant. Please put the answer within \\boxed{}.<|im_end|>
<|im_start|>user
A restaurant has 40 tables with 4 legs and 50 tables with 3 legs. Calculate the total number of legs the restaurant's tables have.<|im_end|>
<|im_start|>assistant
"""

    print("\n" + "="*50)
    print("Running inference...")
    print("="*50)

    output = generate_text(model, prompt, max_new_tokens=100)

    print("\n" + "="*50)
    print("OUTPUT:")
    print("="*50)
    print(output)


if __name__ == "__main__":
    main()