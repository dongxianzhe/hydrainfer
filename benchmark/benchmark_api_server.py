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

def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    # return a list of tuple
    # first str is prompt
    # second is prompt length
    # third is output length

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # some of these will be filtered out, so sample more than we need
    sampled_indices = random.sample(range(len(dataset)),
                                    int(num_requests * 1.2))
    dataset = [dataset[i] for i in sampled_indices]

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            # This is because TGI causes errors when the input or output length
            # is too short.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests

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