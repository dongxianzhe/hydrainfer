from transformers import PreTrainedTokenizerBase
from typing import List, Tuple
import json
import random

def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]: # prompt, prompt_length, output_length
    # return a list of tuple
    # first str is prompt
    # second is prompt length
    # third is output length

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # some of these will be filtered out, so sample more than we need
    sampled_indices = random.sample(range(len(dataset)),
                                    int(num_requests * 1.2))
    dataset = [dataset[i] for i in sampled_indices]

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            # This is because TGI causes errors when the input or output length
            # is too short.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests

# if __name__ == '__main__':
#     from transformers import AutoTokenizer
#     tokenizer = AutoTokenizer.from_pretrained('gpt2')
#     input_requests = sample_sharegpt_requests(
#         dataset_path='./dataset/ShareGPT_V3_unfiltered_cleaned_split.json',
#         num_requests=3,
#         tokenizer=tokenizer)
#     for request in input_requests:
#         print(request[0])
#         print(request[1])
#         print(request[2])
#         print('====================================================================================================')