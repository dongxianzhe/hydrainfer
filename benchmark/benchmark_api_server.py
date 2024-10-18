import time
import aiohttp
from tqdm import tqdm
import json
from transformers import PreTrainedTokenizerBase, AutoTokenizer
from dataclasses import dataclass, field
import random
from typing import List, Tuple, AsyncGenerator, Optional
import argparse
import numpy as np
import asyncio
from backend_request_func import RequestFuncInput, RequestFuncOutput, request_func
from sample_dataset import sample_sharegpt_requests

async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)

async def benchmark(
    model_id: str, 
    api_url: str,  
    best_of: int, 
    use_beam_search: bool, 
    input_requests: List[Tuple[str, int, int]], # (prompt, prompt length, output_lenght)
    request_rate: float
):
    pbar = tqdm(total=len(input_requests))
    benchmark_start_time = time.perf_counter()
    tasks = []
    async for request in get_request(input_requests, request_rate):
        prompt, prompt_len, output_len = request
        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
            best_of=best_of,
            use_beam_search=use_beam_search,
        )
        tasks.append(asyncio.create_task(request_func(request_func_input=request_func_input,pbar=pbar)))
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)
    pbar.close()
    benchmark_duration = time.perf_counter() - benchmark_start_time
    # todo compute metrics

def main(args: argparse.Namespace):
    # 1. set seed
    # 2. load dataset and sample request
    # 3. run async request
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    input_requests = sample_sharegpt_requests(dataset_path='./dataset/ShareGPT_V3_unfiltered_cleaned_split.json', num_requests=args.num_prompts, tokenizer=tokenizer)

    asyncio.run(benchmark(
        input_requests=input_requests,         
        request_rate = args.request_rate
    ))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark the online serving throughput.")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)