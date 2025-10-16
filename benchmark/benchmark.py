import os
import json
import time
import random
import asyncio
import argparse
import numpy as np
from tqdm import tqdm
from typing import AsyncGenerator
from dataclasses import dataclass, field, asdict
from metric import OnlineRequestOutput, BenchmarkResult, MethodResults, Statistics, make_statistic
from synthetic_dataset import SyntheticDataset, SyntheticDataEntry
from backend import get_server_proxy
from timestamp import get_intervals

def analyze_result(args: argparse.Namespace, method_results: MethodResults):
    pbar = tqdm(range(sum([len(result.outputs) for result in method_results.results])), desc="analyzing result")
    for result in method_results.results:
        for output in result.outputs:
            pbar.update(1)
            if not output.success:
                continue
            output.total_tokens = len(output.token_times)
            output.latency = output.token_times[-1] - output.start_time
            output.ttft = output.token_times[0] - output.start_time
            output.tbts = sorted([output.token_times[i] - output.token_times[i - 1] for i in range(1, len(output.token_times))])
            output.tbt_statistics = make_statistic(output.tbts)
            output.tpot = output.tbt_statistics.mean
            result.total_tokens += output.total_tokens
            result.total_success += output.success
            result.latencies.append(output.latency)
            result.ttfts.append(output.ttft)
            result.tpots.append(output.tpot)
            result.tbts.extend(output.tbts)
        result.latencies.sort()
        result.ttfts.sort()
        result.tpots.sort()
        result.token_throughput = result.total_tokens / (result.end_time - result.start_time)
        result.request_throughput = result.total_success / (result.end_time - result.start_time)
        result.latency_statistics = make_statistic(result.latencies)
        result.ttft_statistics = make_statistic(result.ttfts)
        result.tpot_statistics = make_statistic(result.tpots)
        result.tbt_statistics = make_statistic(result.tbts)

def log_result(args: argparse.Namespace, method_results: MethodResults):
    # we do not log images to speed up log time
    for result in method_results.results:
        for i, output in enumerate(result.outputs):
            output.entry.images = None
            if i < args.show_result:
                print(f'show request {i} result: {repr(output.output_text)}') # repr is used to not show escaped char
    
    with open(args.result_path, "w") as file:
        json.dump(asdict(method_results), fp=file, indent=4)

from typing import Iterable
async def request_generator(
    dataset: Iterable,
    intervals: Iterable
) -> AsyncGenerator[tuple[int, SyntheticDataEntry], None]:
    for i, request in enumerate(dataset):
        yield i, request
        await asyncio.sleep(intervals[i])


def async_wrapper(func):
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))
    return wrapper


async def benchmark(args: argparse.Namespace, dataset: SyntheticDataset, request_rate: float) -> BenchmarkResult:
    request_rate_scaled = request_rate * args.request_rate_num_requests_scale
    num_requests_scaled = int(args.num_requests * request_rate * args.request_rate_num_requests_scale)
    send_pbar = tqdm(total = num_requests_scaled, desc='send')
    recv_pbar = tqdm(total = num_requests_scaled, desc='recv')
    server_proxy = get_server_proxy(args.backend)
    
    intervals = get_intervals(method=args.request_rate_method, request_rate=request_rate_scaled)
    start_time = time.perf_counter()
    tasks = []
    async for (i, entry) in request_generator(
        dataset=dataset[:num_requests_scaled], 
        intervals=intervals, 
    ):
        if args.only_text:
            entry.images = []
            entry.images_size = []
        tasks.append(asyncio.create_task(server_proxy(args.model_path, entry, send_pbar=send_pbar, recv_pbar=recv_pbar, base_url=f"http://{args.host}:{args.port}/v1", timeout=args.timeout)))
    outputs: list[OnlineRequestOutput] = await asyncio.gather(*tasks)
    end_time = time.perf_counter()
    recv_pbar.close()
    assert len(outputs) > 0

    return BenchmarkResult(
        total_request = num_requests_scaled, 
        request_rate_method = args.request_rate_method, 
        start_time = start_time, 
        end_time = end_time, 
        request_rate = request_rate, 
        outputs = outputs, 
    )


@async_wrapper
async def benchmarks(args: argparse.Namespace, dataset: SyntheticDataset) -> MethodResults:
    results: list[BenchmarkResult] = []
    for request_rate in args.request_rate:
        print(f'start test request rate {request_rate} scale {args.request_rate_num_requests_scale}')
        result = await benchmark(args, dataset, request_rate)
        results.append(result)
    return MethodResults(
        method_name = args.method_name, 
        datasets = dataset.get_dataset_name(), 
        model = args.model, 
        model_path = args.model_path, 
        results = results, 
    )


def main(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)
    dataset = SyntheticDataset.from_cli_args(args, int(args.num_requests * args.request_rate_num_requests_scale * max(args.request_rate)))
    method_results = benchmarks(args, dataset)
    analyze_result(args, method_results)
    log_result(args, method_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmarking script for inference system', conflict_handler='resolve')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(f'--model', type=str, default="llava-hf/llava-1.5-7b-hf", help='The name of the model.')
    parser.add_argument(f'--model-path', type=str, default="llava-hf/llava-1.5-7b-hf", help='The name of the model.')
    parser.add_argument(
        "--backend",
        type=str,
        # choices=["ours", "vllm", 'tgi', 'sglang'],
        default="unknown", 
        required=False, 
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
        default=[float('10')], 
        metavar='rate',
        help="Number of requests per second"
        "we use poisson process or real work load to synthetic"
        "the request arrival times.",
    )
    parser.add_argument(
        "--request-rate-num-requests-scale",
        type=int,
        default=1,
        help="scaling factor of num requests and request scale, set to the number of gpu.",
    )
    parser.add_argument(
        "--request-rate-method", 
        type=str,
        choices=['poisson', 'azure_code', 'azure_conv', 'burstgpt', 'mooncake'], 
        default="poisson", 
        help="choose the request rate sampling method", 
    )
    SyntheticDataset.add_cli_args(parser)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--result-path", type=str)
    parser.add_argument("--method-name", type=str)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--show-result", type=int, default=0, help='show some inference result to stdout')
    parser.add_argument("--only_text", type=int, default=0, help="if set true, benchmark only send prompt of multimodal request")
    args, remain_args = parser.parse_known_args()
    print(f'benchmark args {args}')
    main(args)