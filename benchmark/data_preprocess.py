import os
import json
from datasets import load_dataset, Dataset
from dataclasses import dataclass, field, asdict
import argparse


@dataclass
class ProcessedRequestResult:
    prompt: list[str] = field(default_factory=list)
    images_size: list[tuple[int, int]] = field(default_factory=list) 
    # input_text: list[str] = field(default_factory=list)
    # input_tokens: list[int] = field(default_factory=list)
    n_input_tokens: list[int] = field(default_factory=list)
    # output_tokens: list[int] = field(default_factory=list)
    n_output_tokens: list[int] = field(default_factory=list)
    output: list[str] = field(default_factory=list)


@dataclass
class PreProcessedResult:
    dataset: str
    model_path: str
    requests: list[ProcessedRequestResult] = field(default_factory=list)


def preprocess_dataset(path: str, split: str, model_path: str) -> Dataset:
    cache_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        'cache', 
        f"{path}_{split}.parquet".replace('/', '-')
    )
    cache_log = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        'cache', 
        f"{path}_{split}.log".replace('/', '-')
    )
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    if os.path.exists(cache_file):
        print(f"Loading preprocessed dataset from {cache_file}")
        dataset = Dataset.from_parquet(cache_file)
    else:
        # load raw data
        print(f"Loading raw dataset and preprocessing {cache_file}")
        dataset = load_dataset(path=path, split=split)

        # process data
        from vllm import LLM, SamplingParams
        n_requests = len(dataset)

        result = PreProcessedResult(
            dataset = path, 
            model_path = model_path, 
            requests=[ProcessedRequestResult() for _ in range(n_requests)], 
        )
        requests: list = []
        sampling_params: list[SamplingParams] = []
        for i in range(n_requests):
            entry = dataset[i]
            result.requests[i].prompt = entry['question']
            result.requests[i].images_size = [entry['image'].size]
            requests.append({
                "prompt": f"USER: <image>\n{entry['question']}\nASSISTANT:", 
                "multi_modal_data": {
                    "image" : entry['image'], 
                }, 
            })
            sampling_params.append(SamplingParams(
                temperature=0, 
                max_tokens=1024, 
                ignore_eos=False, 
            ))
        entrypoint = LLM(model=model_path, max_model_len=4096)
        outputs = entrypoint.generate(requests, sampling_params=sampling_params)

        for i in range(n_requests):
            # results[i].input_text = outputs[i].prompt
            # results[i].input_tokens = outputs[i].prompt_token_ids
            result.requests[i].n_input_tokens = len(outputs[i].prompt_token_ids)
            # results[i].output_tokens = outputs[i].outputs[0].token_ids
            result.requests[i].n_output_tokens = len(outputs[i].outputs[0].token_ids) 
            result.requests[i].output = outputs[i].outputs[0].text
        with open(cache_log, "w") as file:
            json.dump(asdict(result), fp=file, indent=4)
        
        dataset = dataset.add_column('max_tokens', [request.n_output_tokens for request in result.requests])
        # save data
        dataset.to_parquet(cache_file)
        print(f"Preprocessed dataset saved to {cache_file}")
    return dataset


def load_processed_dataset(path: str, split: str, model_path: str) -> Dataset:
    cache_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        'cache', 
        f"{path}_{split}.parquet".replace('/', '-')
    )
    if os.path.exists(cache_file):
        print(f"Loading preprocessed dataset from {cache_file}")
        dataset = Dataset.from_parquet(cache_file)
    else:
        raise Exception(f'preprocessed dataset not exists {cache_file}')
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='preprocess dataset', conflict_handler='resolve')
    parser.add_argument(f'--model-path', type=str, default="llava-hf/llava-1.5-7b-hf", help='The name of the model.')
    parser.add_argument(f'--dataset', type=str, default="llava-hf/llava-1.5-7b-hf", help='The name of the dataset.')
    args, remain_args = parser.parse_known_args()
    dataset = preprocess_dataset(args.dataset, 'test', args.model_path)