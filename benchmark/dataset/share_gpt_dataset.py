import json
from dataclasses import dataclass
import random

@dataclass
class ShareGPTDataEntry:
    prompt: str
    output_text: str

class ShareGPTDataset:
    """
    usage:
        dataset = ShareGPTDataset(dataset_path='ShareGPT_V3_unfiltered_cleaned_split.json')
        n_samples: random sample n_samples data entry from dataset
    """
    def __init__(self, dataset_path: str, max_words: int=1024, n_samples: int=-1):
        with open(dataset_path) as f:
            self.dataset = json.load(f)
        # filter out the conversations with less than 2 turns.
        self.dataset = [data for data in self.dataset if len(data["conversations"]) >= 2]
        # only keep the first two turns of each conversation.
        self.dataset = [ShareGPTDataEntry(
            prompt=data["conversations"][0]["value"], 
            output_text=data["conversations"][1]["value"]
        ) for data in self.dataset]

        if max_words > 0:
            self.dataset = [data for data in self.dataset if len(data.prompt.split()) <= max_words]

        if n_samples > 0:
            assert n_samples <= len(self.dataset), f"n_samples {n_samples} is greater than total {len(self.dataset)}"
            sampled_indices = random.sample(range(len(self.dataset)), n_samples)
            self.dataset = [self.dataset[i] for i in sampled_indices]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> ShareGPTDataEntry:
        return self.dataset[index]