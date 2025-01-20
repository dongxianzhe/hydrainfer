import random
from PIL import Image
import argparse
from benchmark_metric import BenchmarkMetricsBuilder
from transformers import AutoProcessor, AutoTokenizer
from dataset.share_gpt_dataset import ShareGPTDataEntry, ShareGPTDataset
from dataset.simulated_dataset import SimulatedDataset, SimulatedDataEntry


def vllm_benchmark(dataset: SimulatedDataset, output_text_lens: list[int]): 
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.vllm_model_name, max_model_len=4096, enforce_eager=True)

    sampling_params = []

    metric_builder = BenchmarkMetricsBuilder()
    inputs = []
    if args.dataset == 'sim':
        for i, entry in enumerate(dataset):
            inputs.append({
                "prompt": entry.prompt, 
                "multi_modal_data": {
                    "image" : entry.image, 
                }, 
            })
            sampling_params.append(SamplingParams(temperature=0, max_tokens=output_text_lens[i], ignore_eos=True))
    elif args.dataset == 'sharegpt':
        for i, entry in enumerate(dataset):
            inputs.append({
                "prompt": entry.prompt, 
            })
            sampling_params.append(SamplingParams(temperature=0, max_tokens=output_text_lens[i], ignore_eos=True))
    else:
        raise Exception(f'unsupported dataset {args.dataset} for vllm backend')
        
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


def dxz_benchmark(dataset, output_text_lens: list[int], args: argparse.Namespace):
    from dxz.entrypoint.mllm import MLLM, MLLMConfig
    from dxz.request.request import Request, SamplingParameters

    requests: list[Request] = []
    if args.dataset == 'sim':
        requests = [Request(
            request_id = i, 
            prompt = entry.prompt, 
            image = entry.image, 
            image_base64 = None, 
            sampling_params = SamplingParameters(
                max_tokens = output_text_lens[i], 
            ), 
        ) for i, entry in enumerate(dataset)]
    elif args.dataset == 'sharegpt':
        requests = [Request(
            request_id = i, 
            prompt = entry.prompt, 
            image = None, 
            image_base64 = None, 
            sampling_params = SamplingParameters(
                max_tokens = output_text_lens[i], 
            ), 
        )for i, entry in enumerate(dataset)]
    else:
        raise Exception(f'unsupported dataset {args.dataset} for dxz backend')

    config = MLLMConfig.from_cli_args(args)
    print(config)
    mllm = MLLM(config)
    metric_builder = BenchmarkMetricsBuilder()
    outputs = mllm.generate(requests)
    for output in outputs:
        metric_builder.append(
            input_len = 1, 
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
    # 1. load dataset
    if args.dataset == 'sharegpt':
        dataset = ShareGPTDataset(dataset_path='./dataset/ShareGPT_V3_unfiltered_cleaned_split.json')[:args.num_prompts]
        if args.output_input:
            for entry in dataset:
                print(entry.prompt)
    elif args.dataset == 'sim':
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        image_path = f'./dataset/cherry_blossom.jpg'
        if args.has_images == 'one':
            has_images=[True for _ in range(args.num_prompts)]
        elif args.has_images == 'zero':
            has_images=[False for _ in range(args.num_prompts)]
        elif args.has_images == 'random':
            has_images=[random.choice([False, True]) for _ in range(args.num_prompts)]
        else:
            raise Exception(f'invalid has images args {has_images}')
        print(f'has_images {has_images}')
        prompt_text_lens = [random.randint(args.prompt_min_tokens, args.prompt_max_tokens) for _ in range(args.num_prompts)]
        print(f'prompt_text_lens: {prompt_text_lens}')
        dataset = SimulatedDataset(tokenizer, image_path, has_images, prompt_text_lens)
        if args.output_input:
            for entry in dataset:
                print(entry.prompt)
    else:
        raise Exception(f'invalid dataset {args.dataset}')

    output_text_lens = [random.randint(args.decode_min_tokens, args.decode_max_tokens) for _ in range(args.num_prompts)]
    print(f'output_text_lens: {output_text_lens}')
    # 2. generate
    if args.backend == 'vllm':
        vllm_benchmark(dataset, output_text_lens)
    elif args.backend == 'dxz':
        dxz_benchmark(dataset, output_text_lens, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark the offline serving.")
    parser.add_argument(
        "--dataset", 
        type=str,
        choices=["sharegpt", "sim"],
        default="sim",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=100,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        '--output-input',
        action='store_true',
        default=False,
        help='output prompt text'
    )
    parser.add_argument(
        '--output-text',
        action='store_true',
        default=False,
        help='output generated text'
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="seed",
    )
    parser.add_argument(
        "--prompt-min-tokens",
        type=int,
        default=8,
        help="min number of prefill prompt tokens not include image",
    )
    parser.add_argument(
        "--prompt-max-tokens",
        type=int,
        default=8,
        help="max number of prefill prompt tokens not include image",
    )
    parser.add_argument(
        "--decode-min-tokens",
        type=int,
        default=3,
        help="min number of tokens generated.",
    )
    parser.add_argument(
        "--decode-max-tokens",
        type=int,
        default=10,
        help="max number of tokens genreated.",
    )
    parser.add_argument(
        "--has-images", 
        type=str,
        choices=["one", "zero", "random"],
        default="one",
    )
    parser.add_argument(
        "--tokenizer-path", 
        type=str,
        help="used to get tokenizer used in simulated dataset",
        default="llava-hf/llava-1.5-7b-hf", 
    )
    parser.add_argument(
        "--backend", 
        type=str,
        choices=["vllm", "dxz"],
        default="vllm",
    )
    parser.add_argument(
        "--vllm-model-name", 
        type=str,
        help="vllm backend model name",
        default="llava-hf/llava-1.5-7b-hf", 
    )
    try:
        from dxz.entrypoint.mllm import MLLMConfig
        parser = MLLMConfig.add_cli_args(parser)
    except Exception as e:
        print(e)
        pass
        
    args = parser.parse_args()
    main(args)