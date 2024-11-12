from PIL import Image
import argparse
import time
import os


image_path = f'./dataset/cherry_blossom.jpg'
image = Image.open(image_path)
question = "What is the content of this image?"
prompt = f"USER: <image>\n{question}\nASSISTANT:"

def main(args: argparse.Namespace):
    # 1. prepare input
    inputs = [{
        "prompt" : prompt, 
        "multi_modal_data":{
            "image": image
        },
    } for _ in range(args.num_prompts)]

    # 2. generate
    if args.backend == 'vllm':
        from vllm import LLM, SamplingParams
        llm = LLM(model="llava-hf/llava-1.5-7b-hf", max_model_len=4096)
        sampling_params = SamplingParams(temperature=0, max_tokens=50)

        start = time.perf_counter()
        outputs = llm.generate(inputs, sampling_params=sampling_params)
        end = time.perf_counter()

    elif args.backend == 'dxz':
        from dxz.engine.mllm_engine import MLLMEngine, MLLMEngineConfig
        config = MLLMEngineConfig()
        llm = MLLMEngine(config)

        start = time.perf_counter()
        outputs = llm.generate(inputs)
        end = time.perf_counter()

    # 3. show result
    if args.output_text:
        for output in outputs:
            print(output)
    print(f'total time {end - start: 3}')

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