import os
import json
import time
import random
import asyncio
import argparse
from tabulate import tabulate
import numpy as np
from tqdm import tqdm
from openai import AsyncOpenAI
from typing import AsyncGenerator
from dataclasses import dataclass, field, asdict
from transformers import AutoTokenizer
from metric import OnlineRequestOutput, BenchmarkResult, BenchmarkMetrics, BenchmarkMetricsBuilder, MethodResults
from synthetic_dataset import SyntheticDataset, SyntheticDataEntry
from backend import get_server_proxy


def log_result(args: argparse.Namespace, dataset: SyntheticDataset, method_results: MethodResults):
    # we do not log images to speed up log time
    for result in method_results.results:
        for output in result.outputs:
            output.entry.images = None
    
    with open(args.result_path, "w") as file:
        json.dump(asdict(method_results), fp=file, indent=4)


async def poisson_process_request_generator(
    dataset,
    request_rate: float,
) -> AsyncGenerator[tuple[int, SyntheticDataEntry], None]:
    for i, request in enumerate(dataset):
        yield i, request
        if request_rate == float('inf'):
            continue
        interval = np.random.exponential(1.0 / request_rate)
        await asyncio.sleep(interval)


def async_wrapper(func):
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapper


async def benchmark(args: argparse.Namespace, dataset: SyntheticDataset, client: AsyncOpenAI, request_rate: float) -> BenchmarkResult:
    send_pbar = tqdm(total = len(dataset), desc='send')
    recv_pbar = tqdm(total = len(dataset), desc='recv')
    server_proxy = get_server_proxy(args.backend)

    start_time = time.perf_counter()
    tasks = []
    async for (i, entry) in poisson_process_request_generator(dataset=dataset, request_rate=request_rate):
        tasks.append(asyncio.create_task(server_proxy(args.model_path, entry, send_pbar=send_pbar, recv_pbar=recv_pbar, client=client)))
    outputs: list[OnlineRequestOutput] = await asyncio.gather(*tasks)
    end_time = time.perf_counter()
    recv_pbar.close()
    assert len(outputs) > 0

    return BenchmarkResult(
        start_time = start_time, 
        end_time = end_time, 
        request_rate = request_rate, 
        outputs = outputs, 
    )


@async_wrapper
async def benchmarks(args: argparse.Namespace, dataset: SyntheticDataset) -> MethodResults:
    openai_api_key = "EMPTY"
    openai_api_base = f"http://{args.host}:{args.port}/v1"
    client = AsyncOpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    results: list[BenchmarkResult] = []
    for request_rate in args.request_rate:
        print(f'start test request rate {request_rate}')
        result = await benchmark(args, dataset, client, request_rate)
        results.append(result)
    return MethodResults(
        method_name = args.method_name, 
        model=args.model_path, 
        results = results, 
    )


def main(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)
    dataset = SyntheticDataset(
        num_requests = args.num_requests, 
        textcaps     = args.textcaps, 
        pope         = args.pope, 
        mme          = args.mme, 
        text_vqa     = args.text_vqa, 
        vizwiz_vqa   = args.vizwiz_vqa, 
    )                 
    method_results = benchmarks(args, dataset)
    log_result(args, dataset, method_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmarking script for inference system', conflict_handler='resolve')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(f'--model', type=str, default="llava-hf/llava-1.5-7b-hf", help='The name of the model.')
    parser.add_argument(f'--model-path', type=str, default="llava-hf/llava-1.5-7b-hf", help='The name of the model.')
    parser.add_argument(
        "--backend",
        type=str,
        choices=["ours", "vllm", 'tgi', 'sglang'],
        required=True, 
        help="choose the backend"
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=10,
        help="Number of requests to process.",
    )
    parser.add_argument(
        '--request-rate', 
        type=float, 
        nargs='*', 
        default=[float('inf')], 
        metavar='rate',
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--textcaps", type=int, default=int(os.environ.get("TEXTCAPS", 0)))
    parser.add_argument("--pope", type=int, default=int(os.environ.get("POPE", 0)))
    parser.add_argument("--mme", type=int, default=int(os.environ.get("MME", 0)))
    parser.add_argument("--text_vqa", type=int, default=int(os.environ.get("TEXT_VQA", 0)))
    parser.add_argument("--vizwiz_vqa", type=int, default=int(os.environ.get("VIZWIZ_VQA", 0)))
    parser.add_argument("--result-path", type=str)
    parser.add_argument("--method-name", type=str)
    args, remain_args = parser.parse_known_args()
    print(f'benchmark args {args}')
    main(args)