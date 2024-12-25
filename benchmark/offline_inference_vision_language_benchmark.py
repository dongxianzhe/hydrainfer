import random
import numpy as np
import time
from PIL import Image
import argparse
import time
import os
from benchmark_metric import BenchmarkMetrics, BenchmarkMetricsBuilder
from simulated_dataset import SimulatedDataset
from transformers import AutoProcessor

model_name = "llava-hf/llava-1.5-7b-hf"
image_path = f'./dataset/cherry_blossom.jpg'
image = Image.open(image_path)
question = "What is the content of this image?"
prompt = f"USER: <image>\n{question}\nASSISTANT:"

def vllm_benchmark(dataset: SimulatedDataset): 
    from vllm import LLM, SamplingParams
    llm = LLM(model=model_name, max_model_len=4096, enforce_eager=True)

    sampling_params = [SamplingParams(temperature=0, max_tokens=input['max_tokens'], ignore_eos=True) for input in dataset]

    metric_builder = BenchmarkMetricsBuilder()
    outputs = llm.generate(dataset, sampling_params=sampling_params)

    for output in outputs:
        metric_builder.append(
            input_len = len(output.prompt_token_ids), 
            success = True, 
            output_len = len(output.outputs[0].token_ids), 
            arrival_time = output.metrics.arrival_time, 
            finished_time = output.metrics.finished_time, 
            token_times = output.metrics.token_times if hasattr(output.metrics, "token_times") else [], 
        )
    metrics = metric_builder.get_metrics()
    metrics.print()

    if args.output_text:
        for output in outputs:
            print(output.outputs[0].text)

def dxz_benchmark(dataset: SimulatedDataset):
    import torch
    from dxz.engine.engine import EngineConfig, Engine, SchedulerConfig
    from dxz.memory.virtual_kv_cache import MemoryConfig
    from dxz.memory.compiler import CompilerConfig
    config = EngineConfig(
        model_name = model_name, 
        dtype = torch.half, 
        device = torch.device('cuda:0'), 
        memory_config=MemoryConfig(
            memory_management_policy='vanilla', 
            num_blocks = 25000, 
            block_size = 16, 
        ), 
        multi_threads_forward=True, 
        multi_streams_forward=True, 
        scheduler_config=SchedulerConfig(
            batch_policy = 'continuousbatch', 
            priority='decode', 
            max_running_sequences = 10, 
            max_batch_fill_tokens = 1024, 
            max_batch_embed_images= 3, 
            batch_embed_fill=False,
            debug_mode=args.debug, 
        ), 
        compiler_config=CompilerConfig(
            max_tokens = 64, 
            disaggregate_embed_prefill = True, 
            kv_cache_eviction_policy = None, 
            window_size = 28, 
            attention_sink_size = 4, 
            token_pruning_policy = None, 
            n_embed_output_tokens = 460,
        ), 
        multi_thread_request_process=True, 
        batch_image_embed_forward=True, 
    )
    engine = Engine(config)

    metric_builder = BenchmarkMetricsBuilder()
    outputs = engine.generate(dataset)

    for output in outputs:
        metric_builder.append(
            input_len = output.input_len, 
            success = True, 
            output_len = len(output.token_times), 
            arrival_time = output.arrival_time, 
            finished_time = output.finished_time, 
            token_times = output.token_times, 
            )

    metrics = metric_builder.get_metrics()
    metrics.print()

    if args.output_text:
        for output in outputs:
            print(output.text)

def main(args: argparse.Namespace):
    random.seed(args.seed)

    if args.only_prefill:
        dataset = SimulatedDataset(
            processor=AutoProcessor.from_pretrained(model_name), 
            image_path=image_path, 
            has_images=[True for _ in range(args.num_prompts)], 
            prompt_text_lens = [17 for i in range(args.num_prompts)], 
            output_text_lens = [1 for i in range(args.num_prompts)]
            )
    else:
        output_text_lens = [random.randint(args.min_tokens, args.max_tokens) for _ in range(args.num_prompts)]
        print(output_text_lens)
        dataset = SimulatedDataset(
            processor=AutoProcessor.from_pretrained(model_name), 
            image_path=image_path, 
            has_images=[True for _ in range(args.num_prompts)], 
            prompt_text_lens = [17 for i in range(args.num_prompts)], 
            output_text_lens = output_text_lens
            )

    # 2. generate
    if args.backend == 'vllm':
        vllm_benchmark(dataset)
    elif args.backend == 'dxz':
        dxz_benchmark(dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark the online serving throughput.")
    parser.add_argument(
        "--backend", 
        type=str,
        choices=["vllm", "dxz"],
        default="vllm",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=100,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        '--output-text',
        action='store_true',
        default=False,
        help='output generated text'
    )

    parser.add_argument(
        '--only-prefill',
        action='store_true',
        default=False,
        help='only test prefill performance'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        default=False,
        help='print scheduler step info'
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="seed",
    )
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