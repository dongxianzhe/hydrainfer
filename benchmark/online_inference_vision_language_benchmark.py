import base64
from openai import OpenAI, AsyncOpenAI
from typing import AsyncGenerator, Optional
from transformers import AutoProcessor
import time
import random
import numpy as np
import argparse
import asyncio
from tqdm import tqdm
from simulated_dataset import SimulatedDataset
from benchmark_metric import BenchmarkMetricsBuilder
from dataclasses import dataclass, field

@dataclass
class RequestInput:
    prompt: str = ""
    image_base64: str = ""
    max_tokens: int = 0


@dataclass
class RequestOutput:
    prompt: str = ""
    success: bool = False
    input_len: int = 0
    output_text: str = ""
    start_time: float = 0.
    token_times: list[float] = field(default_factory=list)



openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8888/v1"
client = AsyncOpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

prompt = "What is the content of this image?"
image_path = f'./dataset/cherry_blossom.jpg'
model_name = "llava-hf/llava-1.5-7b-hf"
image_path = f'./dataset/cherry_blossom.jpg'

async def vllm_request_func(input: RequestInput, pbar: Optional[tqdm] = None) -> RequestOutput:
    output = RequestOutput()
    response = await client.chat.completions.create(
        messages=[{
            "role":"user",
            "content": [
                {
                    "type": "text",
                    "text": input.prompt, 
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{input.image_base64}"
                    },
                },
            ],
        }],
        model=model_name,
        max_tokens = input.max_tokens, 
        temperature=0., 
        stream=True, 
    )
    output.success = True
    output.start_time = time.perf_counter()
    async for chunk in response:
        content = chunk.choices[0].delta.content
        output.output_text += content
        output.token_times.append(time.perf_counter())

    output.prompt = input.prompt
    if pbar:
        pbar.update(1)

    return output

async def dxz_request_func(input: RequestInput, pbar: Optional[tqdm] = None) -> RequestOutput:
    response = await client.chat.completions.create(
        messages=[{
            "role":"user",
            "content" : input.prompt, 
            "image" : input.image_base64, 
        }],
        model=model_name,
        max_tokens = input.max_tokens, 
        stream=True, 
    )
    output = RequestOutput()
    output.success = True
    output.start_time = time.perf_counter()
    async for chunk in response:
        content = chunk.choices[0].delta.content
        output.output_text += content + " "
        output.token_times.append(time.perf_counter())

    output.prompt = input.prompt
    if pbar:
        pbar.update(1)

    return output

async def benchmark(dataset, args: argparse.Namespace):
    pbar = tqdm(total=len(dataset))
    start = time.perf_counter()

    async def get_request(
        dataset,
        request_rate: float,
    ) -> AsyncGenerator[tuple[str, int, int], None]:
        for request in iter(dataset):
            yield request
            if request_rate == float('inf'):
                continue
            interval = np.random.exponential(1.0 / request_rate)
            await asyncio.sleep(interval)

    metric_builder = BenchmarkMetricsBuilder()
    tasks = []
    async for request in get_request(dataset, args.request_rate):
        request_func_input = RequestInput(prompt=request.prompt, image_base64=request.image_base64, max_tokens=request.max_tokens)
        if args.backend == 'vllm':
            tasks.append(asyncio.create_task(vllm_request_func(request_func_input, pbar=pbar)))
        elif args.backend == 'dxz':
            tasks.append(asyncio.create_task(dxz_request_func(request_func_input, pbar=pbar)))
        else:
            raise Exception(f'invalid backend {args.backend}')

    outputs: list[RequestOutput] = await asyncio.gather(*tasks)

    pbar.close()
    for output in outputs:
        metric_builder.append(
            input_len = output.input_len, 
            success = output.success, 
            output_len = len(output.token_times), 
            arrival_time = output.start_time, 
            finished_time = output.token_times[-1], 
            token_times = output.token_times,
        )
    metrics = metric_builder.get_metrics()
    metrics.print()

    if args.output_text:
        for i, output in enumerate(outputs):
            print(f'{output.prompt} {output.output_text}')

def main(args: argparse.Namespace):
    # 1. set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    # 2. sample dataset
    output_text_lens = [random.randint(args.min_tokens, args.max_tokens) for _ in range(args.num_prompts)]
    print(output_text_lens)
    dataset = SimulatedDataset(
        processor=AutoProcessor.from_pretrained(model_name), 
        image_path=image_path, 
        has_images=[True for _ in range(args.num_prompts)], 
        prompt_text_lens = [17 for i in range(args.num_prompts)], 
        output_text_lens = output_text_lens
        )
    # 3. async request
    asyncio.run(benchmark(
        dataset, 
        args=args, 
    ))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark the online serving throughput.")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--backend", 
        type=str,
        choices=["vllm", "dxz"],
        default="vllm",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=10,
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
    parser.add_argument(
        '--output-text',
        action='store_true',
        default=False,
        help='output generated text'
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=3,
        help="min number of tokens generated.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10,
        help="max number of tokens genreated.",
    )
    args = parser.parse_args()

    main(args)