import concurrent
import os
from tqdm import tqdm
import argparse
import json
import datasets
from dataclasses import dataclass, asdict
from synthetic_dataset import SyntheticDataEntry, encode_base64_content_from_image, SyntheticSourceDataset


def get_preprocessed_data_path(path: str) -> str:
    return os.path.join(path, 'synthetic_source_dataset.json')


def get_limit_max_tokens() -> int:
    return 1024


def process_entry(i: int, dataset: list[dict], outputs: list[dict], args: argparse.Namespace) -> SyntheticDataEntry:
    """
    This function processes each entry and returns a SyntheticDataEntry.
    
    Args:
        i (int): The index for the current entry.
        dataset (List[dict]): A list of datasets containing 'question' and 'image'.
        outputs (List[dict]): A list of output objects containing token information and text.
        args (object): The arguments containing dataset and model information.
    
    Returns:
        SyntheticDataEntry: The processed data entry with the appropriate fields.
    """
    return SyntheticDataEntry(
        prompt=dataset[i]['question'], 
        images=[encode_base64_content_from_image(dataset[i]['image'])], 
        images_size=[dataset[i]['image'].size], 
        dataset=args.dataset, 
        n_output_tokens_ref=len(outputs[i].outputs[0].token_ids), 
        output_ref=outputs[i].outputs[0].text, 
        n_input_tokens_ref=len(outputs[i].prompt_token_ids)
    )


def preprocess(args: argparse.Namespace):
    from vllm import LLM, SamplingParams
    dataset = datasets.load_dataset(args.dataset_path, split='test')
    n_requests = min(len(dataset), args.n_requests)
    requests: list = [{
        "prompt": f"USER: <image>\n{dataset[i]['question']}\nASSISTANT:", 
        "multi_modal_data": {
            "image" : dataset[i]['image'], 
        }} for i in tqdm(range(n_requests), desc="Processing requests input")]
    sampling_params: list[SamplingParams] = [SamplingParams(
            temperature=0, 
            max_tokens=get_limit_max_tokens(), 
            ignore_eos=False, 
        ) for i in range(n_requests)]
    entrypoint = LLM(model=args.model_path)
    outputs = entrypoint.generate(requests, sampling_params=sampling_params)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_entry, 
                                        range(n_requests), 
                                        [dataset] * n_requests, 
                                        [outputs] * n_requests, 
                                        [args] * n_requests),
                            total=n_requests, 
                            desc="Processing synthetic source dataset"))

    synthetic_source_dataset = SyntheticSourceDataset(
        dataset=args.dataset, 
        model=args.model,
        dataset_path=args.dataset_path, 
        model_path=args.model_path, 
        entries=results
    )

    synthetic_source_dataset.entries = [entry for entry in synthetic_source_dataset.entries if entry.n_output_tokens_ref != get_limit_max_tokens()]
    print(f'Discard requests {len(synthetic_source_dataset.entries) - n_requests} that exceed the length')
    with open(get_preprocessed_data_path(args.dataset_path), "w") as file:
        json.dump(asdict(synthetic_source_dataset), fp=file, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data preprocessor')
    parser.add_argument(f'--dataset', type=str, default="lmms-lab/TextCaps", help='The name of the dataset in huggingface.')
    parser.add_argument(f'--dataset-path', type=str, default="/datasets/lmms-lab/TextCaps", help='The path of the dataset in local.')
    parser.add_argument(f'--model', type=str, default="llava-hf/llava-1.5-7b-hf", help='The name of the model in huggingface.')
    parser.add_argument(f'--model-path', type=str, default="/models/llava-1.5-7b-hf", help='The path of the model in local.')
    parser.add_argument(f'--n-requests', type=int, default=100000000, help='number of requests processed from dataset')
    args = parser.parse_args()
    preprocess(args)