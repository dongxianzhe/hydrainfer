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
from stream_completions_backend_request_func import RequestFuncInput, RequestFuncOutput, request_func
from sample_dataset import sample_sharegpt_requests

@dataclass
class StreamCompletionsMetrics:
    completed: int    # completed number of request
    total_input: int  # total input number of tokens
    total_output: int # total output number of tokens
    request_throughput: float # completed / total time
    input_throughput: float   # total_input / total time
    output_throughput: float  # total_output / total time
    mean_input_len: int
    median_input_len: int
    max_input_len: int
    mean_output_len: int
    median_output_len: int
    max_output_len: int
    mean_latency_ms: float
    median_latency_ms: float
    p90_latency_ms: float
    p99_latency_ms: float
    mean_ttft_ms: float
    median_ttft_ms: float
    p90_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    p90_tpot_ms: float
    p99_tpot_ms: float

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
    inputs: List[Tuple[str, int, int]], # (prompt, prompt length, output_lenght)
    request_rate: float
):
    pbar = tqdm(total=len(inputs))
    benchmark_start_time = time.perf_counter()
    tasks = []
    async for request in get_request(inputs, request_rate):
        prompt, prompt_len, output_len = request
        request_func_input = RequestFuncInput(prompt=prompt)
        tasks.append(asyncio.create_task(request_func(request_func_input=request_func_input,pbar=pbar)))
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)
    pbar.close()
    benchmark_duration = time.perf_counter() - benchmark_start_time
    # compute metrics
    input_lens = []
    # output_lens = [input[2] for input in inputs]
    actual_output_lens = []
    total_input = 0
    completed = 0
    tpots = []
    ttfts = []
    latency = []
    for i in range(len(outputs)):
        if outputs[i].success:
            actual_output_lens.append(len(outputs[i].itl) + 1)
            input_lens.append(inputs[i][1])
            total_input += inputs[i][1]
            for tpot in outputs[i].itl:
                tpots.append(tpot)
            ttfts.append(outputs[i].ttft)
            latency.append(outputs[i].latency)
            completed += 1
        else:
            actual_output_lens.append(0)
    
    metrics = StreamCompletionsMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        mean_input_len=np.mean(input_lens),
        median_input_len=np.median(input_lens),
        max_input_len=max(input_lens),
        mean_output_len=np.mean(actual_output_lens),
        median_output_len=np.median(actual_output_lens),
        max_output_len=max(actual_output_lens),
        request_throughput=completed / benchmark_duration,
        input_throughput=total_input / benchmark_duration,
        output_throughput=sum(actual_output_lens) / benchmark_duration,
        mean_latency_ms=np.mean(latency) * 1000,
        median_latency_ms=np.median(latency) * 1000,
        p90_latency_ms=np.percentile(latency, 90) * 1000,
        p99_latency_ms=np.percentile(latency, 99) * 1000,
        mean_ttft_ms=np.mean(ttfts or 0) * 1000,  # ttfts is empty if streaming is not supported by backend
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        p90_ttft_ms=np.percentile(ttfts or 0, 90) * 1000,
        p99_ttft_ms=np.percentile(ttfts or 0, 99) * 1000,
        mean_tpot_ms=np.mean(tpots) * 1000,
        median_tpot_ms=np.median(tpots) * 1000,
        p90_tpot_ms=np.percentile(tpots, 90) * 1000 if len(tpots) > 0 else np.nan,
        p99_tpot_ms=np.percentile(tpots, 99) * 1000 if len(tpots) > 0 else np.nan,
    )

    print("{s:{c}^{n}}".format(s=' Serving Benchmark Result ', n=50, c='='))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):",
                                    benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:",
                                 metrics.total_output))
    print("{:<40} {:<10}".format("Mean input tokens:", metrics.mean_input_len))
    print("{:<40} {:<10}".format("Median input tokens:", metrics.median_input_len))
    print("{:<40} {:<10}".format("Max input tokens:", metrics.max_input_len))
    print("{:<40} {:<10}".format("Mean generated tokens:",
                                 metrics.mean_output_len))
    print("{:<40} {:<10}".format("Median generated tokens:",
                                 metrics.median_output_len))
    print("{:<40} {:<10}".format("Max generated tokens:",
                                 metrics.max_output_len))
    print("{s:{c}^{n}}".format(s='Throughput', n=50, c='-'))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):",
                                    metrics.request_throughput))
    print("{:<40} {:<10.2f}".format("Input token throughput (tok/s):",
                                    metrics.input_throughput))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):",
                                    metrics.output_throughput))
    print("{s:{c}^{n}}".format(s='Time to Latency', n=50, c='-'))
    print("{:<40} {:<10.2f}".format("Mean Latency (ms):", metrics.mean_latency_ms))
    print("{:<40} {:<10.2f}".format("Median Latency (ms):", metrics.median_latency_ms))
    print("{:<40} {:<10.2f}".format("P90 Latency (ms):", metrics.p90_latency_ms))
    print("{:<40} {:<10.2f}".format("P99 Latency (ms):", metrics.p99_latency_ms))
    print("{s:{c}^{n}}".format(s='Time to First Token', n=50, c='-'))
    print("{:<40} {:<10.2f}".format("Mean TTFT (ms):", metrics.mean_ttft_ms))
    print("{:<40} {:<10.2f}".format("Median TTFT (ms):",
                                    metrics.median_ttft_ms))
    print("{:<40} {:<10.2f}".format("P90 TTFT (ms):", metrics.p90_ttft_ms))
    print("{:<40} {:<10.2f}".format("P99 TTFT (ms):", metrics.p99_ttft_ms))
    print("{s:{c}^{n}}".format(s='Time per Output Token (excl. 1st token)',
                               n=50,
                               c='-'))
    print("{:<40} {:<10.2f}".format("Mean TPOT (ms):", metrics.mean_tpot_ms))
    print("{:<40} {:<10.2f}".format("Median TPOT (ms):",
                                    metrics.median_tpot_ms))
    print("{:<40} {:<10.2f}".format("P90 TPOT (ms):", metrics.p90_tpot_ms))
    print("{:<40} {:<10.2f}".format("P99 TPOT (ms):", metrics.p99_tpot_ms))
    print("=" * 50)

def main(args: argparse.Namespace):
    # 1. set seed
    # 2. load dataset and sample request
    # 3. run async request
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    inputs = sample_sharegpt_requests(dataset_path='./dataset/ShareGPT_V3_unfiltered_cleaned_split.json', num_requests=args.num_prompts, tokenizer=tokenizer)

    asyncio.run(benchmark(
        inputs=inputs, 
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