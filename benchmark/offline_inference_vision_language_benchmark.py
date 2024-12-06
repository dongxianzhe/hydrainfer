import numpy as np
import time
from PIL import Image
import argparse
import time
import os
from offline_inference_benchmark import BenchmarkMetrics


image_path = f'./dataset/cherry_blossom.jpg'
image = Image.open(image_path)
question = "What is the content of this image?"
# a prompt with image first
prompt = f"USER: <image>\n{question}\nASSISTANT:"

# a prompt with image second
# prompt = f"USER: \n{question}<image>\nASSISTANT:"

# a prompt with image first after decoded
# prompt = f"USER: <image>\n{question}\nASSISTANT:The image features a tall tower with a spire, surrounded by a beautiful cherry blossom tree. The tree is filled with pink flowers, creating a picturesque scene. The tower stands tall in the background, with the blossoming tree in the foreground. The combination of the tower and the tree creates a captivating and serene atmosphere."

def main(args: argparse.Namespace):
    # 1. prepare input
    inputs = [{
        "prompt" : prompt, 
        "multi_modal_data":{
            "image": image
        }, 
        "max_tokens": (i + 1) * 10,
    } for i in range(args.num_prompts)]

    # 2. generate
    if args.backend == 'vllm':
        from vllm import LLM, SamplingParams
        llm = LLM(model="llava-hf/llava-1.5-7b-hf", max_model_len=4096, enforce_eager=True)

        sampling_params = [SamplingParams(temperature=0, max_tokens=input['max_tokens'] + 1, ignore_eos=True) for input in inputs]

        start = time.perf_counter()
        outputs = llm.generate(inputs, sampling_params=sampling_params)
        end = time.perf_counter()
        duration = end - start
        completed = len(outputs)
        input_lens = []
        latencies = []
        ttfts = []
        tpots = []
        output_lens = []
        for output in outputs:
            input_lens.append(len(output.prompt_token_ids))
            latencies.append(output.metrics.finished_time - output.metrics.arrival_time)
            ttfts.append(output.metrics.first_token_time - output.metrics.first_scheduled_time)
            if hasattr(output.metrics, "token_times"):
                for i in range(1, len(output.metrics.token_times)):
                    tpot = output.metrics.token_times[i] - output.metrics.token_times[i - 1]
                    tpots.append(tpot)
            output_lens.append(len(output.outputs[0].token_ids))
        metrics = BenchmarkMetrics(
            benchmark_duration=duration, 
            completed=completed,
            total_input=sum(input_lens),
            total_output=sum(output_lens),
            mean_input_len=np.mean(input_lens),
            median_input_len=np.median(input_lens),
            max_input_len=max(input_lens),
            mean_output_len=np.mean(output_lens),
            median_output_len=np.median(output_lens),
            max_output_len=max(output_lens),
            request_throughput=completed / duration,
            input_throughput=sum(input_lens) / duration,
            output_throughput=sum(output_lens) / duration,
            mean_latency_ms=np.mean(latencies) * 1000,
            median_latency_ms=np.median(latencies) * 1000,
            p90_latency_ms=np.percentile(latencies, 90) * 1000,
            p99_latency_ms=np.percentile(latencies, 99) * 1000,
            mean_ttft_ms=np.mean(ttfts or 0) * 1000,
            median_ttft_ms=np.median(ttfts or 0) * 1000,
            p90_ttft_ms=np.percentile(ttfts or 0, 90) * 1000,
            p99_ttft_ms=np.percentile(ttfts or 0, 99) * 1000,
            mean_tpot_ms=np.mean(tpots) * 1000,
            median_tpot_ms=np.median(tpots) * 1000,
            p90_tpot_ms=np.percentile(tpots, 90) * 1000 if len(tpots) > 0 else np.nan,
            p99_tpot_ms=np.percentile(tpots, 99) * 1000 if len(tpots) > 0 else np.nan,
        )
        metrics.print()

        if args.output_text:
            for output in outputs:
                print(output.outputs[0].text)

    elif args.backend == 'dxz':
        import torch
        from dxz.engine.engine import EngineConfig, Engine, SchedulerConfig
        from dxz.memory.virtual_kv_cache import MemoryConfig
        from dxz.memory.compiler import CompilerConfig
        config = EngineConfig(
            model_name = "llava-hf/llava-1.5-7b-hf", 
            dtype = torch.half, 
            device = torch.device('cuda:0'), 
            memory_config=MemoryConfig(
                num_blocks = 20000, 
                block_size = 16, 
            ), 
            scheduler_config=SchedulerConfig(
                batch_policy = 'continuousbatch', 
                max_running_sequences = 10, 
            ), 
            compiler_config=CompilerConfig(
                max_tokens = 64, 
                kv_cache_eviction_policy = None, 
                window_size = 28, 
                attention_sink_size = 4, 
                token_pruning_policy = None, 
                n_embed_output_tokens = 64, 
            ), 
        )
        engine = Engine(config)

        start = time.perf_counter()

        outputs = engine.generate(inputs)

        end = time.perf_counter()
        duration = end - start
        completed = len(outputs)
        input_lens = []
        latencies = []
        ttfts = []
        tpots = []
        output_lens = []
        for output in outputs:
            input_lens.append(output.input_len)
            latencies.append(output.latency)
            ttfts.append(output.ttft)
            tpots += output.tpot
            output_lens.append(len(output.tpot) + 1)

        metrics = BenchmarkMetrics(
            benchmark_duration=duration, 
            completed=completed,
            total_input=sum(input_lens),
            total_output=sum(output_lens),
            mean_input_len=np.mean(input_lens),
            median_input_len=np.median(input_lens),
            max_input_len=max(input_lens),
            mean_output_len=np.mean(output_lens),
            median_output_len=np.median(output_lens),
            max_output_len=max(output_lens),
            request_throughput=completed / duration,
            input_throughput=sum(input_lens) / duration,
            output_throughput=sum(output_lens) / duration,
            mean_latency_ms=np.mean(latencies) * 1000,
            median_latency_ms=np.median(latencies) * 1000,
            p90_latency_ms=np.percentile(latencies, 90) * 1000,
            p99_latency_ms=np.percentile(latencies, 99) * 1000,
            mean_ttft_ms=np.mean(ttfts or 0) * 1000,
            median_ttft_ms=np.median(ttfts or 0) * 1000,
            p90_ttft_ms=np.percentile(ttfts or 0, 90) * 1000,
            p99_ttft_ms=np.percentile(ttfts or 0, 99) * 1000,
            mean_tpot_ms=np.mean(tpots) * 1000,
            median_tpot_ms=np.median(tpots) * 1000,
            p90_tpot_ms=np.percentile(tpots, 90) * 1000 if len(tpots) > 0 else np.nan,
            p99_tpot_ms=np.percentile(tpots, 99) * 1000 if len(tpots) > 0 else np.nan,
        )
        metrics.print()
        if args.output_text:
            for output in outputs:
                print(output.text)


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
    args = parser.parse_args()

    main(args)