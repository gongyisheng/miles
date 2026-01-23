#!/usr/bin/env python3
"""
Test inference with torch_dist checkpoint using Megatron.
This loads the checkpoint properly and runs inference to verify it works.

Usage: torchrun --nproc_per_node=1 test_torch_dist_inference.py \
    --tokenizer-model /path/to/hf/model \
    --load /path/to/torch_dist \
    --hf-checkpoint /path/to/hf/model
"""

import os
import sys
import torch

# Add miles to path
miles_path = os.path.expanduser("~/Documents/dev/miles")
if os.path.exists(miles_path):
    sys.path.insert(0, miles_path)
else:
    miles_path = "/root/miles"
    if os.path.exists(miles_path):
        sys.path.insert(0, miles_path)


def get_args():
    from megatron.training.arguments import parse_args, validate_args
    from miles.backends.megatron_utils.arguments import set_default_megatron_args

    def add_test_args(parser):
        parser.add_argument("--hf-checkpoint", type=str, required=True)
        parser.add_argument("--test-prompt", type=str,
                          default="What is 2 + 3? Let me solve this step by step.")
        try:
            parser.add_argument("--padded-vocab-size", type=int, default=None)
        except:
            pass
        return parser

    args = parse_args(add_test_args)
    args = set_default_megatron_args(args)

    # Minimal settings for inference
    args.micro_batch_size = 1
    args.global_batch_size = 1

    validate_args(args)
    return args


def main():
    import torch.distributed as dist
    from megatron.core.enums import ModelType
    from megatron.training.training import get_model
    from megatron.training.checkpointing import load_checkpoint
    from transformers import AutoTokenizer

    # Initialize distributed
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    global_rank = int(os.getenv("RANK", "0"))

    torch.cuda.set_device(local_rank)
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("RANK", str(global_rank))
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12356")

    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=global_rank,
        device_id=torch.device(f"cuda:{local_rank}"),
    )

    args = get_args()

    from miles.backends.megatron_utils.initialize import init
    from miles.backends.megatron_utils.model_provider import get_model_provider_func

    init(args)

    print(f"\n{'='*60}")
    print("Creating model...")
    print(f"{'='*60}")

    model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=False)
    print(f"Model created: {type(model)}")

    # Print model structure
    if hasattr(model[0], 'module'):
        print(f"Model module keys: {list(model[0].module.state_dict().keys())[:10]}")

    print(f"\n{'='*60}")
    print("Loading checkpoint...")
    print(f"{'='*60}")

    # Load checkpoint
    iteration, num_floating_point_operations_so_far = load_checkpoint(model, None, None)
    print(f"Loaded checkpoint at iteration {iteration}")

    # Get the actual model
    megatron_model = model[0]
    if hasattr(megatron_model, 'module'):
        megatron_model = megatron_model.module

    megatron_model.eval()

    print(f"\n{'='*60}")
    print("Running inference test...")
    print(f"{'='*60}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)

    # Test prompt
    prompt = args.test_prompt
    print(f"Prompt: {prompt}")

    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    print(f"Input shape: {input_ids.shape}")
    print(f"Input tokens: {input_ids[0].tolist()[:20]}...")

    # Generate tokens one by one
    max_new_tokens = 50
    generated = input_ids.clone()

    with torch.no_grad():
        for i in range(max_new_tokens):
            # Get position ids
            seq_len = generated.shape[1]
            position_ids = torch.arange(seq_len, device=generated.device).unsqueeze(0)

            # Forward pass - Megatron model interface
            # The model expects (input_ids, position_ids, attention_mask)
            attention_mask = torch.ones_like(generated)

            # Megatron forward
            output = megatron_model(
                input_ids=generated,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )

            # Get logits for last token
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            next_token_logits = logits[:, -1, :]

            # Greedy decoding
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append
            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

            # Print progress
            if i < 10 or i % 10 == 0:
                decoded = tokenizer.decode(next_token[0])
                print(f"  Token {i}: {next_token.item()} -> '{decoded}'")

    # Decode full output
    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)

    print(f"\n{'='*60}")
    print("Generated output:")
    print(f"{'='*60}")
    print(output_text)

    # Also compare with HuggingFace model
    print(f"\n{'='*60}")
    print("Comparing with HuggingFace model...")
    print(f"{'='*60}")

    from transformers import AutoModelForCausalLM

    hf_model = AutoModelForCausalLM.from_pretrained(
        args.hf_checkpoint,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).cuda().eval()

    with torch.no_grad():
        hf_output = hf_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    hf_text = tokenizer.decode(hf_output[0], skip_special_tokens=True)
    print("HuggingFace output:")
    print(hf_text)

    # Compare
    print(f"\n{'='*60}")
    print("Comparison:")
    print(f"{'='*60}")
    if output_text == hf_text:
        print("MATCH! Outputs are identical.")
    else:
        print("MISMATCH! Outputs differ.")
        print(f"Megatron length: {len(output_text)}")
        print(f"HF length: {len(hf_text)}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()