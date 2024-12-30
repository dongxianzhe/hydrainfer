from dataclasses import dataclass, field
from typing import List, Optional
import aiohttp
from tqdm import tqdm
import time
import json
import asyncio
import traceback
import sys

@dataclass
class RequestFuncInput:
    prompt: str

@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0
    ttft: float = 0  # Time to first token
    itl: List[float] = field(default_factory=list)  # List of inter-token latencies
    error: str = ""

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

async def request_func(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {
            'Content-Type': 'application/json'
        }
        payload = {
            "model": "gpt2",
            "prompt": request_func_input.prompt, 
            "max_tokens": 50,
            "temperature": 0, 
            "stream": True,
        }
        output = RequestFuncOutput()
        most_recent_timestamp = start_timestamp = time.perf_counter()
        try:
            async with session.post(url='http://127.0.0.1:8888/v1/completions', json=payload, headers=headers) as response:
                if response.status == 200:
                    async for chunk, _ in response.content.iter_chunks():
                        # eg. b'data: {"id":"cmpl-5b2c049fe2664f2ca7f12915b5e4a8f9","object":"text_completion","created":1729229567,"model":"gpt2","choices":[{"index":0,"text":",","logprobs":null,"finish_reason":null,"stop_reason":null}],"usage":null}\n\n'
                        # eg. b'data: [DONE]\n\n'
                        chunk = chunk.strip() # remove \n\n
                        chunk = chunk[6:] # remove data: 
                        if not chunk or chunk == b'[DONE]':
                            pass
                        else:
                            timestamp = time.perf_counter()
                            if output.ttft == 0:
                                output.ttft = timestamp - start_timestamp
                            else:
                                output.itl.append(timestamp - most_recent_timestamp)
                            most_recent_timestamp = timestamp
                            data = json.loads(chunk.decode("utf-8")) # convert to json
                            output.generated_text += data['choices'][0]['text']
                    latency = time.perf_counter() - start_timestamp
                    output.success = True
                    output.latency = latency
        except:
            print(chunk)
            output.success = False
            print(traceback.format_exc())
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))
        
    if pbar:
        pbar.update(1)
    return output

# async def main():
#     input = RequestFuncInput(
#         prompt = "Hello",
#     )

#     output = await request_func(input)
#     print(input.prompt + output.generated_text)
#     print(f'------------------------------------------------------------------------------------------')
#     print(output)

# if __name__ == '__main__':
#     asyncio.run(main())