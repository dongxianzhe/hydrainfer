import random
import torch
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

    sampling_params = []

    metric_builder = BenchmarkMetricsBuilder()
    inputs = []
    for request in dataset:
        inputs.append({
            "prompt": request.prompt, 
            "multi_modal_data": {
                "image" : request.image, 
            }, 
            "max_tokens": request
        })
        sampling_params.append(SamplingParams(temperature=0, max_tokens=request.max_tokens, ignore_eos=True))
        
    outputs = llm.generate(inputs, sampling_params=sampling_params)

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

def dxz_benchmark(dataset: SimulatedDataset, args: argparse.Namespace):
    from dxz.engine.engine import EngineConfig
    from dxz.entrypoint.mllm import MLLM
    config = EngineConfig.from_cli_args(args)
    print(config)

    mllm = MLLM(config)

    metric_builder = BenchmarkMetricsBuilder()
    outputs = mllm.generate(dataset)

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
        dxz_benchmark(dataset, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark the offline serving.")
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
    try:
        from dxz.engine.engine import EngineConfig
        parser = EngineConfig.add_cli_args(parser)
    except:
        pass

    args = parser.parse_args()
    main(args)