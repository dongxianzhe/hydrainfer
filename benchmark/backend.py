import time
import argparse
from tqdm import tqdm
from openai import AsyncOpenAI
from metric import OnlineRequestOutput
from synthetic_dataset import SyntheticDataEntry


async def our_server_proxy(model_path: str, entry: SyntheticDataEntry, send_pbar: tqdm, recv_pbar: tqdm, client: AsyncOpenAI) -> OnlineRequestOutput:
    send_pbar.update(1)
    output = OnlineRequestOutput(entry=entry)
    output.start_time = time.perf_counter()
    response = await client.chat.completions.create(
        messages = [{
            "role" : "user", 
            "content" : f"<image>\n{entry.prompt}\n", 
            "image" : entry.images[0],
        }], 
        max_tokens=1024, 
        model = model_path,
        stream=True, 
    )
    output.success = True
    async for chunk in response:
        context = chunk.choices[0].delta.content
        if output.output_text != "":
            output.output_text += " "
        output.output_text += context
        output.token_times.append(time.perf_counter())
    output.prompt = entry.prompt
    recv_pbar.update(1)
    return output


async def vllm_server_proxy(model_path: str, entry: SyntheticDataEntry, send_pbar: tqdm, recv_pbar: tqdm, client: AsyncOpenAI) -> OnlineRequestOutput:
    send_pbar.update(1)
    output = OnlineRequestOutput(entry=entry)
    output.start_time = time.perf_counter()
    response = await client.chat.completions.create(
        messages = [{
            "role":"user",
            "content": [
                {
                    "type": "text",
                    "text": entry.prompt, 
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{entry.images[0]}"
                    },
                },
            ],
        }], 
        max_tokens=128, 
        model=model_path,
        temperature=0., 
        stream=True, 
    )
    output.success = True
    async for chunk in response:
        context = chunk.choices[0].delta.content
        if isinstance(context, str):
            output.output_text += context
        output.token_times.append(time.perf_counter())
    output.prompt = entry.prompt
    recv_pbar.update(1)
    return output


def get_server_proxy(backend: str):
    if backend == 'ours':
        return our_server_proxy
    if backend in ['vllm', 'tgi']:
        return vllm_server_proxy
    if backend == 'sglang':
        return vllm_server_proxy
    raise Exception('invalid server proxy')