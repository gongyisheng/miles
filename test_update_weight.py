import requests
import json
import hashlib
import pickle
import torch

from sglang.srt.utils import MultiprocessingSerializer

base_dir = "/tmp"
uuids = ["3c0a80c3", "be6553d1"]

def _tensor_hash(t: torch.Tensor) -> str:
    """Compute SHA256 hash of tensor data."""
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy()).hexdigest()

def make_update_weight_from_tensor_payload(_uuid):
    f = f"{base_dir}/flattened_tensor_0_{_uuid}.pt"
    tensor = torch.load(f, map_location='cuda', weights_only=True)
    print(f"tensor hash: {_tensor_hash(tensor)}")
    f = f"{base_dir}/metadata_0_{_uuid}.pkl"
    metadata = pickle.load(open(f, "rb"))
    flattened_tensor_data = {
        "flattened_tensor": tensor,
        "metadata": metadata,
    }
    serialized_tensors = []
    serialized_data = MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True)
    serialized_tensors.append(serialized_data)

    kwargs = {
        "serialized_named_tensors": serialized_tensors,
        "load_format": "flattened_bucket",
        "flush_cache": False,
        "weight_version": "1"
    }
    return kwargs

def send_request(url, payload):
    print(f"send request to {url}")
    payload_hash = hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    print(f"[DEBUG] payload hash: {payload_hash}")

    try:
        response = requests.post(
            url,
            json=payload
        )

        print(f"Status: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        return response.json()
    except Exception as e:
        print("error")
        return {}

def main():
    print("start")

    url="http://localhost:15000/release_memory_occupation"
    payload={}
    send_request(url, payload)
    print("request0")

    url = "http://localhost:15000/resume_memory_occupation"
    payload={'tags': ['weights']}
    send_request(url, payload)
    print("request1")

    url = "http://localhost:15000/update_weights_from_tensor"
    payload = make_update_weight_from_tensor_payload(uuids[0])
    send_request(url, payload)
    print("request2")

    url = "http://localhost:15000/update_weights_from_tensor"
    payload = make_update_weight_from_tensor_payload(uuids[1])
    print("request3")

if __name__ == "__main__":
    main()