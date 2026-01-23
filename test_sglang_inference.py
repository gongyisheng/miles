import requests
from transformers import AutoTokenizer


def main():
    host = "localhost"
    port = 15000
    tokenizer_path = "/root/Qwen2.5-0.5B-Instruct/"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # Same prompt as test_megatron_inference.py
    prompt = """<|im_start|>system
You are a helpful assistant. Please put the answer within \\boxed{}.<|im_end|>
<|im_start|>user
A restaurant has 40 tables with 4 legs and 50 tables with 3 legs. Calculate the total number of legs the restaurant's tables have.<|im_end|>
<|im_start|>assistant
"""

    input_ids = tokenizer.encode(prompt, add_special_tokens=False)

    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 1,
            "max_new_tokens": 512,
            "skip_special_tokens": True,
            "no_stop_trim": True,
            "spaces_between_special_tokens": False,
        },
        "return_logprob": True,
    }

    print(f"Sending request to http://{host}:{port}/generate")
    print(f"Input tokens: {len(input_ids)}")
    print(f"Prompt:\n{prompt}")
    print("=" * 50)

    response = requests.post(f"http://{host}:{port}/generate", json=payload)

    if response.status_code == 200:
        result = response.json()
        print("Response:")
        print(result.get("text", result))
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    main()
