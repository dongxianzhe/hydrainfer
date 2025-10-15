import httpx
import time
import traceback
import json
import time
import traceback
from tqdm import tqdm
from metric import OnlineRequestOutput
from synthetic_dataset import SyntheticDataEntry


def prepare_openai_compatible_payload(model_path: str, entry: SyntheticDataEntry) -> dict:
    content = [
        {
            "type": "text",
            "text": entry.prompt, 
        },
    ]
    for image in entry.images:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image}"
            },
        })
    payload = {
        "model": model_path,
        "messages": [
            {
                "role": "user",
                "content": content, 
            }
        ],
        "max_tokens": entry.n_output_tokens_ref,
        "temperature": 0.0,
        "stream": True,
    }
    return payload

async def send_request(payload: dict, entry: SyntheticDataEntry, base_url: str="http://localhost:8000/v1", timeout: float=60) -> OnlineRequestOutput:
    output = OnlineRequestOutput(entry=entry)
    output.start_time = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", f"{base_url}/chat/completions", json=payload) as response:
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    if line.strip() == "data: [DONE]":
                        break
                    data = json.loads(line[len("data: "):])
                    delta = data["choices"][0]["delta"].get("content")
                    if isinstance(delta, str):
                        output.output_text += delta
                    output.token_times.append(time.perf_counter())

        output.success = True
        output.prompt = entry.prompt
    except Exception:
        output.success = False
        output.error_msg = traceback.format_exc()
    return output

async def openai_compatible_server_proxy(model_path: str, entry: SyntheticDataEntry, send_pbar: tqdm, recv_pbar: tqdm, base_url="http://localhost:8000/v1", timeout: float=60) -> OnlineRequestOutput:
    send_pbar.update(1)
    payload = prepare_openai_compatible_payload(model_path, entry)
    output = await send_request(payload, entry, base_url, timeout)
    recv_pbar.update(1)
    return output

async def vllm_server_proxy(model_path: str, entry: SyntheticDataEntry, send_pbar: tqdm, recv_pbar: tqdm, base_url="http://localhost:8000/v1", timeout: float=60) -> OnlineRequestOutput:
    send_pbar.update(1)
    payload = prepare_openai_compatible_payload(model_path, entry)
    payload['ignore_eos']=True
    output = await send_request(payload, entry, base_url, timeout)
    recv_pbar.update(1)
    return output

def get_server_proxy(backend: str):
    if backend in ['ours', 'tgi', 'sglang']:
        return openai_compatible_server_proxy
    if 'vllm' in backend:
        return vllm_server_proxy
    return openai_compatible_server_proxy