from dataclasses import dataclass, field
from typing import List, Optional
import aiohttp
from tqdm import tqdm
import time
import json

@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    best_of: int = 1
    use_beam_search: bool = False


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0
    ttft: float = 0  # Time to first token
    itl: List[float] = field(default_factory=list)  # List of inter-token latencies
    prompt_len: int = 0
    error: str = ""

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

async def request_func(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        payload = {
            "prompt": request_func_input.prompt,
            "temperature": 0.0,
            "best_of": request_func_input.best_of,
            "max_tokens": request_func_input.output_len,
            "top_k": 50,
            "stream": True,
            "ignore_eos": True,
        }
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len
        generated_text = ""
        ttft = 0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    async for chunk, _ in response.content.iter_chunks:
                        chunk = chunk.strip()
                        if not chunk:
                            continue
                        chunk = chunk[:-1]
                        data = json.loads(chunk.decode("utf-8"))
                        if data['text']:
                            timestamp = time.perf_counter()
                            if ttft == 0:
                                ttft = time.perf_counter - st
                            else:
                                output.itl.append(timestamp - most_recent_timestamp)
                            most_recent_timestamp = timestamp
                            generated_text = data['text']
                    latency = time.perf_counter - st
                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
        except:
            print(chunk)
            output.success = False
        
    if pbar:
        pbar.update(1)
    return output
