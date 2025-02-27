import time
import random
import asyncio
import argparse
import functools
import numpy as np
from tqdm import tqdm
from openai import AsyncOpenAI
from typing import AsyncGenerator
from dataclasses import dataclass, field
from dataset.simulated_dataset import SimulatedDataset, SimulatedDataEntry
from transformers import AutoTokenizer
from benchmark_metric import BenchmarkMetrics, BenchmarkMetricsBuilder

scenarios = [{
    'argument' : '--summary', 
    'help' : "whether add text summary scenario and it's proportion weight in dataset", 
    'default' : 0, 
    'request_generator': lambda: {
        'n_prompt_tokens': random.randint(500, 600), 
        'n_images': 0,
        'n_output_tokens': random.randint(1, 20), 
        'ttft_slo': 2, 
        'tpot_slo': 0.08, 
    }, 
}, {
    'argument' : '--msummary', 
    'help' : "whether add multi-modal summary scenario and it's proportion weight in dataset", 
    'default' : 0, 
    'request_generator': lambda: {
        'n_prompt_tokens': random.randint(500, 600), 
        'n_images': 1, 
        'n_output_tokens': random.randint(1, 20), 
        'ttft_slo': 2,  
        'tpot_slo': 0.08,
    }, 
}, {
    'argument' : '--chat', 
    'help' : "whether add text chat scenario and it's proportion weight in dataset", 
    'default' : 0, 
    'request_generator': lambda: {
        'n_prompt_tokens': random.randint(10, 20), 
        'n_images':0, 
        'n_output_tokens': random.randint(10, 99), 
        'ttft_slo': 1, 
        'tpot_slo': 0.04, 
    }, 
}, {
    'argument' : '--mchat', 
    'help' : "whether add multi-modal chat scenario and it's proportion weight in dataset", 
    'default' : 1, 
    'request_generator': lambda: {
        'n_prompt_tokens': random.randint(10, 20), 
        'n_images': 1, 
        'n_output_tokens': random.randint(10, 99), 
        'ttft_slo': 1,  
        'tpot_slo': 0.04,
    }, 
}]

def prepare_requests(args: argparse.Namespace) -> SimulatedDataset:
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.model_path is None:
        args.model_path = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    dataset = SimulatedDataset(tokenizer, image_path='./dataset/cherry_blossom.jpg')

    population = []
    weights = []
    for scenario in scenarios:
        weight = getattr(args, scenario['argument'].removeprefix('--'))
        if weight > 0:
            population.append(scenario['request_generator'])
            weights.append(weight)
            
    request_generators = random.choices(population=population, weights=weights, k=args.num_requests)
    for i in range(args.num_requests):
        dataset.append(**request_generators[i]())

    return dataset


@dataclass
class BenchmarkResult:
    metric: BenchmarkMetrics
    output_text: list[str] = field(default_factory=list)


def log_result(args: argparse.Namespace, dataset: SimulatedDataset, results: list[BenchmarkResult]):
    if args.log_requests:
        from tabulate import tabulate
        headers = ["request_id", "prompt", "n_prompt_tokens", "n_images",  "n_output_tokens", "ttft_slo", "tpot_slo"]
        data = [
            (i, entry.prompt[:40], entry.n_prompt_tokens, len(entry.images),entry.n_output_tokens, entry.ttft_slo, entry.tpot_slo)
            for i, entry in enumerate(dataset)
        ]
        table = tabulate(data, headers, tablefmt="grid")
        print(table)
    

    for i, result in enumerate(results):
        print(f'==================== benchmark result {i} ====================')
        if args.log_output:
            for i, output_text in enumerate(result.output_text):
                print(f'{i}: {output_text}')
        result.metric.print()

    if len(results) > 1:
        print(f'==================== slo analysis ====================')
        data = [
            ["Alice", 24, "Engineer"],
            ["Bob", 30, "Doctor"],
            ["Charlie", 22, "Artist"]
        ]
        headers = ["request_rate", "ttft_slo_attainment", "tpot_slo_attainment", "slo_attainment"]
        data = []
        for request_rate, result in zip(args.request_rate, results):
            data.append((request_rate, result.metric.ttft_slo_attainment, result.metric.tpot_slo_attainment, result.metric.slo_attainment))
        slo_table = tabulate(data, headers, tablefmt="plain")
        print(slo_table)


def vllm_offline_benchmark(args: argparse.Namespace, dataset: SimulatedDataset) -> list[BenchmarkResult]:
    from vllm import LLM, SamplingParams
    entrypoint = LLM(model=args.model_path, max_model_len=4096, enforce_eager=True)
    sampling_params: list[SamplingParams] = []
    requests: list[dict] = []
    for i, entry in enumerate(dataset):
        requests.append({
            "prompt": entry.prompt, 
            "multi_modal_data": {
                "image" : entry.images, 
            }, 
        })
        sampling_params.append(SamplingParams(
            temperature=0, 
            max_tokens=entry.n_output_tokens, 
            ignore_eos=True, 
        ))
    
    metric_builder = BenchmarkMetricsBuilder()
    metric_builder.start()
    outputs = entrypoint.generate(requests, sampling_params=sampling_params)
    metric_builder.end()
    # for offline inference we only care about throughput so token times are not very important
    for entry, output in zip(dataset, outputs):
        metric_builder.append(
            input_len = entry.n_prompt_tokens, 
            success = output.finished, 
            output_len = len(output.outputs[0].token_ids),
            arrival_time = output.metrics.arrival_time, 
            finished_time = output.metrics.finished_time, 
            token_times = [output.metrics.first_token_time], 
            ttft_slo = entry.ttft_slo,
            tpot_slo = entry.tpot_slo,
        )

    return [BenchmarkResult(
        metric=metric_builder.get_metrics(), 
        output_text=[output.outputs[0].text for output in outputs], 
    )]


def dxz_offline_benchmark(args: argparse.Namespace, dataset: SimulatedDataset) -> list[BenchmarkResult]:
    from dxz.entrypoint import OfflineSingleInstanceEntryPointConfig, OfflineSingleInstanceEntryPoint
    from dxz.request.request import Request, SamplingParameters
    config = OfflineSingleInstanceEntryPointConfig.from_cli_args(args)
    print(config)
    entrypoint = OfflineSingleInstanceEntryPoint(config)
    requests = [Request(
        request_id = i, 
        prompt = entry.prompt, 
        image = entry.images[0] if len(entry.images) > 0 else None,
        image_base64 = None, 
        sampling_params = SamplingParameters(
            max_tokens = entry.n_output_tokens, 
        ), 
    ) for i, entry in enumerate(dataset)]

    metric_builder = BenchmarkMetricsBuilder()
    metric_builder.start()
    outputs = entrypoint.generate(requests)
    metric_builder.end()

    for entry, output in zip(dataset, outputs):
        metric_builder.append(
            input_len = entry.n_prompt_tokens, 
            success = True, 
            output_len = len(output.token_times), 
            arrival_time = output.arrival_time, 
            finished_time = output.finished_time, 
            token_times = output.token_times, 
            ttft_slo = entry.ttft_slo,
            tpot_slo = entry.tpot_slo,
            )

    return [BenchmarkResult(
        metric = metric_builder.get_metrics(), 
        output_text = [output.text for output in outputs],
    )]


async def poisson_process_request_generator(
    dataset,
    request_rate: float,
) -> AsyncGenerator[tuple[int, SimulatedDataEntry], None]:
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


@dataclass
class OnlineRequestOutput:
    entry: SimulatedDataEntry
    prompt: str = ""
    success: bool = False
    output_text: str = ""
    start_time: float = 0.
    token_times: list[float] = field(default_factory=list)


async def vllm_server_proxy(args: argparse.Namespace, entry: SimulatedDataEntry, pbar: tqdm, client: AsyncOpenAI) -> OnlineRequestOutput:
    output = OnlineRequestOutput(entry=entry)
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
                        "url": f"data:image/jpeg;base64,{entry.image_base64[0]}"
                    },
                },
            ],
        }] if len(entry.image_base64) > 0 else [{
            "role":"user",
            "content": [
                {
                    "type": "text",
                    "text": entry.prompt, 
                },
            ],
        }],
        model = args.model_path,
        max_tokens = entry.n_output_tokens, 
        temperature=0., 
        stream=True, 
    )
    output.success = True
    output.start_time = time.perf_counter()
    async for chunk in response:
        context = chunk.choices[0].delta.content
        output.output_text += context
        output.token_times.append(time.perf_counter())
    output.prompt = entry.prompt
    if pbar:
        pbar.update(1)
    return output


async def dxz_server_proxy(args: argparse.Namespace, entry: SimulatedDataEntry, pbar: tqdm, client: AsyncOpenAI) -> OnlineRequestOutput:
    response = await client.chat.completions.create(
        messages = [{
            "role" : "user", 
            "content" : entry.prompt, 
            "image" : entry.image_base64[0]
        }] if len(entry.image_base64) > 0 else [{
            "role" : "user", 
            "content" : entry.prompt, 
        }], 
        model = args.model_name, 
        max_tokens = entry.n_output_tokens, 
        stream = True, 
    )
    output = OnlineRequestOutput(entry=entry)
    output.success = True
    output.start_time = time.perf_counter()
    async for chunk in response:
        content = chunk.choices[0].delta.content
        output.output_text += content + " "
        output.token_times.append(time.perf_counter())
    output.prompt = entry.prompt
    if pbar:
        pbar.update(1)
    return output


async def online_benchmark(args: argparse.Namespace, dataset: SimulatedDataset, server_proxy, client: AsyncOpenAI, request_rate: float) -> BenchmarkResult:
    pbar = tqdm(total = len(dataset))
    metric_builder = BenchmarkMetricsBuilder()

    metric_builder.start()
    tasks = []
    async for (i, entry) in poisson_process_request_generator(dataset=dataset, request_rate=request_rate):
        tasks.append(asyncio.create_task(server_proxy(args, entry, pbar=pbar, client=client)))
    outputs: list[OnlineRequestOutput] = await asyncio.gather(*tasks)

    metric_builder.end()
    pbar.close()
    for output in outputs:
        metric_builder.append(
            input_len = output.entry.n_prompt_tokens, 
            success = output.success, 
            output_len = len(output.token_times), 
            arrival_time = output.start_time, 
            finished_time = output.token_times[-1], 
            token_times = output.token_times, 
            ttft_slo = output.entry.ttft_slo,
            tpot_slo = output.entry.tpot_slo,
        )
        tpot = [output.token_times[i] - output.token_times[i-1] for i in range(1, len(output.token_times))]
        def find_second_largest(nums):
            if len(nums) < 2:
                return 0
            sorted_nums = sorted(nums, reverse=True)
            return sorted_nums[1]
        ttft = output.token_times[0] - output.start_time
        print(f'n_prompt_tok {output.entry.n_prompt_tokens} tpot slo {output.entry.tpot_slo} ttft slo {output.entry.ttft_slo} ttft {ttft} max tpot {max(tpot)} second tpot {find_second_largest(tpot)} tpot {tpot}')
        
    return BenchmarkResult(
        metric = metric_builder.get_metrics(), 
        output_text = [output.output_text for output in outputs]
    )

@async_wrapper
async def online_benchmarks(args: argparse.Namespace, dataset: SimulatedDataset, server_proxy) -> list[BenchmarkResult]:
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8888/v1"
    client = AsyncOpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    results: list[BenchmarkResult] = []
    for request_rate in args.request_rate:
        print(f'start test request rate {request_rate}')
        result = await online_benchmark(args, dataset, server_proxy, client, request_rate)
        results.append(result)
    return results


def benchmark(args: argparse.Namespace, dataset: SimulatedDataset) -> BenchmarkResult:
    benchmark_backends = {
        'offline': {
            'vllm': vllm_offline_benchmark,
            'dxz': dxz_offline_benchmark,
        },
        'online': {
            'vllm': functools.partial(online_benchmarks, server_proxy=vllm_server_proxy),
            'dxz': functools.partial(online_benchmarks, server_proxy=dxz_server_proxy),
        },
    }
    try:
        benchmark_backend = benchmark_backends[args.inference_mode][args.backend]
    except KeyError:
        raise Exception(f"Invalid inference mode or backend: mode={args.inference_mode}, backend={args.backend}")
    return benchmark_backend(args, dataset)


def main(args: argparse.Namespace):
    dataset = prepare_requests(args)
    results = benchmark(args, dataset)
    log_result(args, dataset, results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmarking script for inference system', conflict_handler='resolve')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(f'--model-name', type=str, default="llava-hf/llava-1.5-7b-hf", help='The name of the model.')
    parser.add_argument(f'--model-path', type=str, nargs="?", default=None, help="path to the model, if set none will download model from huggingface to default cache directory of transformers library with the model-name arg.")
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
        "--inference-mode", 
        type=str,
        choices=["offline", "online"],
        default="offline",
    )
    parser.add_argument(
        "--backend", 
        type=str,
        choices=["vllm", "dxz"],
        default="vllm",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument(
        '--log-requests',
        action='store_true',
        default=False,
        help='log request info'
    )
    parser.add_argument(
        '--log-output',
        action='store_true',
        default=False,
        help='log request output'
    )
    for scenario in scenarios:
        parser.add_argument(
            scenario['argument'], 
            type=int,
            default=scenario['default'],
            help=scenario['help'],
        )
    try:
        from dxz.entrypoint import OfflineSingleInstanceEntryPointConfig
        parser = OfflineSingleInstanceEntryPointConfig.add_cli_args(parser)
    except Exception as e:
        print(e)
        print('dxz engine config import failed, ignoring')

    args = parser.parse_args()

    main(args)