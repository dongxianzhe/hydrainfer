import httpx
import time
import traceback
import json
import time
import argparse
import traceback
from tqdm import tqdm
from openai import AsyncOpenAI
from metric import OnlineRequestOutput
from synthetic_dataset import SyntheticDataEntry

async def openai_compatible_server_proxy(model_path: str, entry: SyntheticDataEntry, send_pbar: tqdm, recv_pbar: tqdm, base_url="http://localhost:8000/v1", timeout: float=60) -> OnlineRequestOutput:
    send_pbar.update(1)
    output = OnlineRequestOutput(entry=entry)
    output.start_time = time.perf_counter()
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
    try:
        payload = {
            "model": model_path,
            "messages": [
                {
                    "role": "user",
                    "content": content, 
                }
            ],
            "max_tokens": entry.max_tokens,
            "temperature": 0.0,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=None) as client:
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

    recv_pbar.update(1)
    return output


def get_server_proxy(backend: str):
    if backend == 'ours':
        return openai_compatible_server_proxy
    if backend in ['vllm', 'tgi']:
        return openai_compatible_server_proxy
    if backend == 'sglang':
        return openai_compatible_server_proxy
    return openai_compatible_server_proxy