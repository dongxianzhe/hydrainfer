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
from metric import OnlineRequestOutput, BenchmarkResult, MethodResults, Statistics
from synthetic_dataset import SyntheticDataset, SyntheticDataEntry
from backend import get_server_proxy

def make_statistic(values: list[float]) -> Statistics:
    if len(values) == 0:
        return None
    return Statistics(
        max = max(values), 
        min = min(values), 
        mean = np.mean(values), 
        median = np.median(values), 
        p90 = np.percentile(values, 90), 
        p99 = np.percentile(values, 99), 
        var = np.var(values), 
    )

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
            output.tpots = sorted([output.token_times[i] - output.token_times[i - 1] for i in range(1, len(output.token_times))])
            output.tpot_statistics = make_statistic(output.tpots)
            result.total_tokens += output.total_tokens
            result.total_success += output.success
            result.latencies.append(output.latency)
            result.ttfts.append(output.ttft)
            result.tpots.extend(output.tpots)
        result.latencies.sort()
        result.ttfts.sort()
        result.tpots.sort()
        result.token_throughput = result.total_tokens / (result.end_time - result.start_time)
        result.request_throughput = result.total_success / (result.end_time - result.start_time)
        result.latency_statistics = make_statistic(result.latencies)
        result.ttft_statistics = make_statistic(result.ttfts)
        result.tpot_statistics = make_statistic(result.tpots)

def log_result(args: argparse.Namespace, method_results: MethodResults):
    # we do not log images to speed up log time
    for result in method_results.results:
        for i, output in enumerate(result.outputs):
            output.entry.images = None
            if i < args.show_result:
                print(f'show request {i} result: {repr(output.output_text)}') # repr is used to not show escaped char
    
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


async def benchmark(args: argparse.Namespace, dataset: SyntheticDataset, request_rate: float) -> BenchmarkResult:
    request_rate_scaled = request_rate * args.request_rate_num_requests_scale
    num_requests_scaled = int(args.num_requests * request_rate * args.request_rate_num_requests_scale)
    send_pbar = tqdm(total = num_requests_scaled, desc='send')
    recv_pbar = tqdm(total = num_requests_scaled, desc='recv')
    server_proxy = get_server_proxy(args.backend)
    
    start_time = time.perf_counter()
    tasks = []
    async for (i, entry) in poisson_process_request_generator(
        dataset=dataset[:num_requests_scaled], 
        request_rate=request_rate_scaled, 
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
        datasets = {
            "textcaps" : args.textcaps, 
            "pope" : args.pope, 
            "mme" : args.mme, 
            "text_vqa" : args.text_vqa, 
            "vizwiz_vqa" : args.vizwiz_vqa,
        }, 
        model = args.model, 
        model_path = args.model_path, 
        results = results, 
    )


def main(args: argparse.Namespace):
    random.seed(args.seed)
    np.random.seed(args.seed)
    dataset = SyntheticDataset(
        model_path   = args.model_path, 
        num_requests = int(args.num_requests * args.request_rate_num_requests_scale * max(args.request_rate)), 
        textcaps     = args.textcaps, 
        pope         = args.pope, 
        mme          = args.mme, 
        text_vqa     = args.text_vqa, 
        vizwiz_vqa   = args.vizwiz_vqa, 
    )                 
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
        default=[float('inf')], 
        metavar='rate',
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
    )
    parser.add_argument(
        "--request-rate-num-requests-scale",
        type=int,
        default=1,
        help="scaling factor of num requests and request scale, set to the number of gpu.",
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
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--show-result", type=int, default=0, help='show some inference result to stdout')
    parser.add_argument("--only_text", type=int, default=0, help="if set true, benchmark only send prompt of multimodal request")
    args, remain_args = parser.parse_known_args()
    print(f'benchmark args {args}')
    main(args)