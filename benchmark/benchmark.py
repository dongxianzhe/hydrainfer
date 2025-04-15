import os
import sys
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
from dataset.simulated_dataset import SimulatedDataset, SimulatedDataEntry, SyntheticDataset, encode_base64_content_from_image
from transformers import AutoTokenizer
from benchmark_metric import BenchmarkMetrics, BenchmarkMetricsBuilder
from mme.MMETestPaper import MMETestPaper
from textcaps.TextCapsDataset import TextCapsDataset
from vega.VEGADataset import VEGADataset


scenarios = [{
    'argument' : '--summary', 
    'help' : "whether add text summary scenario and it's proportion weight in dataset", 
    'default' : 0, 
    'request_metadata_generator': lambda: {
        'n_prompt_tokens': random.randint(500, 600), 
        'n_images': 0,
        'n_output_tokens': random.randint(1, 20), 
        'ttft_slo': 2, 
        'tpot_slo': 0.08, 
    }, 
    'is_simulated_scenario': True, 
}, {
    'argument' : '--msummary', 
    'help' : "whether add multi-modal summary scenario and it's proportion weight in dataset", 
    'default' : 0, 
    'request_metadata_generator': lambda: {
        'n_prompt_tokens': random.randint(500, 600), 
        'n_images': 1, 
        'n_output_tokens': random.randint(1, 20), 
        'ttft_slo': 2,  
        'tpot_slo': 0.08,
    }, 
    'is_simulated_scenario': True, 
}, {
    'argument' : '--chat', 
    'help' : "whether add text chat scenario and it's proportion weight in dataset", 
    'default' : 0, 
    'request_metadata_generator': lambda: {
        'n_prompt_tokens': random.randint(10, 20), 
        'n_images':0, 
        'n_output_tokens': random.randint(10, 99), 
        'ttft_slo': 1, 
        'tpot_slo': 0.04, 
    }, 
    'is_simulated_scenario': True, 
}, {
    'argument' : '--mchat', 
    'help' : "whether add multi-modal chat scenario and it's proportion weight in dataset", 
    'default' : 0, 
    'request_metadata_generator': lambda: {
        'n_prompt_tokens': random.randint(10, 20), 
        'n_images': 1, 
        'n_output_tokens': random.randint(10, 99), 
        'ttft_slo': 1,  
        'tpot_slo': 0.04,
    }, 
    'is_simulated_scenario': True, 
}, {
    'argument' : '--mtest', 
    'help' : "whether add test request scenario", 
    'default' : 0, 
    'request_metadata_generator': lambda: {
        'n_prompt_tokens': 9, 
        'n_images': 1, 
        'n_output_tokens': 50, 
        'ttft_slo': 1,  
        'tpot_slo': 0.04,
    }, 
    'is_simulated_scenario': True, 
}, {
    'argument' : '--stall', 
    'help' : "whether add test request scenario", 
    'default' : 0, 
    'request_metadata_generator': lambda: {
        'n_prompt_tokens': random.randint(1, 50), 
        'n_images': random.randint(0, 1), 
        'n_output_tokens': 10, 
        'ttft_slo': 1,  
        'tpot_slo': 0.04,
    }, 
    'is_simulated_scenario': True, 
}, {
    'argument' : '--mme', 
    'help' : "whether add mme dataset", 
    'default' : 0, 
    'request_metadata_generator': lambda: {
        'n_output_tokens': 10, 
        'ttft_slo': 1,  
        'tpot_slo': 0.08,
    }, 
    'is_simulated_scenario': False, 
}, {
    'argument' : '--textcaps', 
    'help' : "whether add textcaps dataset", 
    'default' : 0, 
    'request_metadata_generator': lambda: {
        'n_output_tokens': 100, 
        'ttft_slo': 2,  
        'tpot_slo': 0.08,
    }, 
    'is_simulated_scenario': False, 
}, {
    'argument' : '--vega', 
    'help' : "whether add vega dataset", 
    'default' : 0, 
    'request_metadata_generator': lambda: {
        'n_output_tokens': 100, 
        'ttft_slo': 2,  
        'tpot_slo': 0.08,
    }, 
    'is_simulated_scenario': False, 
}
]

def prepare_requests(args: argparse.Namespace) -> SyntheticDataset:
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.model_path is None:
        args.model_path = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    script_path = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_path, 'dataset', 'cherry_blossom.jpg')
    simulated_dataset = SimulatedDataset(tokenizer, image_path=image_path)

    class SimulatedRequestGenerator:
        def __init__(self, simulated_dataset: SimulatedDataset, metadata_generator):
            self.simulated_dataset = simulated_dataset
            self.metadata_generator = metadata_generator

        def __call__(self) -> SimulatedDataEntry:
            return self.simulated_dataset.make_data_entry(**self.metadata_generator())

    # sample request based simulated scenario weights
    population = []
    weights = []
    for scenario in scenarios:
        scenario_name = scenario['argument'].removeprefix('--')
        weight = getattr(args, scenario_name)
        if weight > 0:
            weights.append(weight)

            is_simulated_scenario = scenario['is_simulated_scenario']
            if is_simulated_scenario:
                population.append(SimulatedRequestGenerator(
                    simulated_dataset=simulated_dataset, 
                    metadata_generator=scenario['request_metadata_generator']
                ))
            else:
                if scenario_name == 'mme':
                    dataset = MMETestPaper(path='mme')
                    class MMEDatasetGenerator:
                        def __init__(self, dataset: MMETestPaper, metadata_generator):
                            self.dataset = dataset
                            self.metadata_generator = metadata_generator

                        def __call__(self) -> SimulatedDataEntry:
                            question = random.choice(self.dataset)
                            prompt = f"<image>\n{question.question}\n"
                            metadata = self.metadata_generator()
                            return SimulatedDataEntry(
                                prompt = prompt, 
                                images = [question.image], 
                                image_base64 = [question.image_base64], 
                                n_prompt_tokens = len(tokenizer.encode(prompt)), 
                                n_output_tokens = metadata['n_output_tokens'], 
                                ttft_slo = metadata['ttft_slo'], 
                                tpot_slo = metadata['tpot_slo'], 
                            )
                    population.append(MMEDatasetGenerator(dataset=dataset, metadata_generator=scenario['request_metadata_generator']))
                if scenario_name == 'textcaps':
                    dataset = TextCapsDataset(path='textcaps')
                    class TextCapsDatasetGenerator:
                        def __init__(self, dataset: TextCapsDataset, metadata_generator):
                            self.dataset = dataset
                            self.metadata_generator = metadata_generator

                        def __call__(self) -> SimulatedDataEntry:
                            question = random.choice(self.dataset)
                            prompt = f"<image>\n{question.question}\n"
                            metadata = self.metadata_generator()
                            return SimulatedDataEntry(
                                prompt = prompt, 
                                images = [question.image], 
                                image_base64 = [question.image_base64], 
                                n_prompt_tokens = len(tokenizer.encode(prompt)), 
                                n_output_tokens = metadata['n_output_tokens'], 
                                ttft_slo = metadata['ttft_slo'], 
                                tpot_slo = metadata['tpot_slo'], 
                            )
                    population.append(TextCapsDatasetGenerator(dataset=dataset, metadata_generator=scenario['request_metadata_generator']))
                if scenario_name == 'vega':
                    dataset = VEGADataset(path='vega')
                    class VEGADatasetGenerator:
                        def __init__(self, dataset: VEGADataset, metadata_generator):
                            self.dataset = dataset
                            self.metadata_generator = metadata_generator

                        def __call__(self) -> SimulatedDataEntry:
                            question = random.choice(self.dataset)
                            prompt = f"<image>{question.context}\n{question.question}\n"
                            metadata = self.metadata_generator()
                            return SimulatedDataEntry(
                                prompt = prompt, 
                                images = [question.image], 
                                image_base64 = [question.image_base64], 
                                n_prompt_tokens = len(tokenizer.encode(prompt)), 
                                n_output_tokens = metadata['n_output_tokens'], 
                                ttft_slo = metadata['ttft_slo'], 
                                tpot_slo = metadata['tpot_slo'], 
                            )
                    dataset_request_generator = VEGADatasetGenerator(dataset=dataset, metadata_generator=scenario['request_metadata_generator'])
                    population.append(dataset_request_generator)
    assert len(weights) > 0, "No scenario is selected"
            
    request_generators = random.choices(population=population, weights=weights, k=args.num_requests)

    synthetic_dataset = SyntheticDataset()
    for i in range(args.num_requests):
        synthetic_dataset.append(request_generators[i]())

    return synthetic_dataset


@dataclass
class BenchmarkResult:
    metric: BenchmarkMetrics
    output_text: list[str] = field(default_factory=list)


def log_result(args: argparse.Namespace, dataset: SyntheticDataset, results: list[BenchmarkResult]):
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
        headers = ["request_rate", "ttft_slo_attainment", "tpot_slo_attainment", "slo_attainment"]
        data = []
        for request_rate, result in zip(args.request_rate, results):
            data.append((request_rate, result.metric.ttft_slo_attainment, result.metric.tpot_slo_attainment, result.metric.slo_attainment))
        slo_table = tabulate(data, headers, tablefmt="plain")
        print(slo_table)


def vllm_offline_benchmark(args: argparse.Namespace, dataset: SyntheticDataset) -> list[BenchmarkResult]:
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


def dxz_offline_benchmark(args: argparse.Namespace, dataset: SyntheticDataset) -> list[BenchmarkResult]:
    import dxz
    from dxz.request import Request, SamplingParameters
    import os
    import hydra
    from hydra import initialize, compose
    from omegaconf import OmegaConf, DictConfig

    config_abspath = os.path.join(dxz.__path__[0], 'config')
    cwd = os.getcwd()
    config_refpath = os.path.relpath(config_abspath, cwd)
    initialize(config_path=config_refpath, job_name='offline benchmark', version_base=None)
    config = compose(config_name='entrypoint')
    config.model.path = args.model_path
    config.model.name = args.model_name
    config.mode = 'offline'
    print(OmegaConf.to_yaml(config))

    from dxz.entrypoint import EntryPoint
    entrypoint = EntryPoint(config) 
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


async def online_benchmark(args: argparse.Namespace, dataset: SyntheticDataset, server_proxy, client: AsyncOpenAI, request_rate: float) -> BenchmarkResult:
    send_pbar = tqdm(total = len(dataset), desc='send')
    recv_pbar = tqdm(total = len(dataset), desc='recv')
    metric_builder = BenchmarkMetricsBuilder()

    start = time.perf_counter()
    metric_builder.start()
    tasks = []
    async for (i, entry) in poisson_process_request_generator(dataset=dataset, request_rate=request_rate):
        tasks.append(asyncio.create_task(server_proxy(args, entry, pbar=recv_pbar, client=client)))
        send_pbar.update(1)
    outputs: list[OnlineRequestOutput] = await asyncio.gather(*tasks)

    metric_builder.end()
    recv_pbar.close()
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
        if args.log_token_times:
            for token_time in output.token_times:
                print(f'output_token_time: {token_time - start}')
        # print(f'n_prompt_tok {output.entry.n_prompt_tokens} tpot slo {output.entry.tpot_slo} ttft slo {output.entry.ttft_slo} ttft {ttft} max tpot {max(tpot)} second tpot {find_second_largest(tpot)} tpot {tpot}')
        
    return BenchmarkResult(
        metric = metric_builder.get_metrics(), 
        output_text = [output.output_text for output in outputs]
    )

@async_wrapper
async def online_benchmarks(args: argparse.Namespace, dataset: SyntheticDataset, server_proxy) -> list[BenchmarkResult]:
    openai_api_key = "EMPTY"
    openai_api_base = f"http://{args.host}:{args.port}/v1"
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


def benchmark(args: argparse.Namespace, dataset: SyntheticDataset) -> BenchmarkResult:
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
        '--log-token-times',
        action='store_true',
        default=False,
        help='log token output times'
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
    args, remain_args = parser.parse_known_args()
    print(f'benchmark args {args}')
    print(f'benchmark remain_args {remain_args}')
    sys.argv[1:] = remain_args

    main(args)