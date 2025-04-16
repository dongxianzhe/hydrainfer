import os
import base64
import random
from datasets import load_dataset
from dataclasses import dataclass
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor


def encode_base64_content_from_image(image: Image.Image) -> str:
    buffered = BytesIO()
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(buffered, format="PNG")
    encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return encoded_string


@dataclass
class SyntheticDataEntry:
    prompt: str
    images: list[str]


class SyntheticDataset:
    def __init__(
        self, 
        num_requests: int,
        textcaps: int = 0, 
        pope: int = 0, 
        mme: int = 0, 
        text_vqa: int = 0, 
        vizwiz_vqa: int = 0, 
    ):
        datasets_metadata = [
            (textcaps, "lmms-lab/TextCaps"), 
            (pope, "lmms-lab/POPE"), 
            (mme, "lmms-lab/MME"), 
            (text_vqa, "lmms-lab/textvqa"), 
            (vizwiz_vqa, "lmms-lab/VizWiz-VQA"), 
        ]

        self.num_requests = num_requests
        datasets = []
        dataset_iters = []
        datasets_name: list[str] = []
        weights: list[int] = []
        for weight, path in datasets_metadata:
            if weight > 0:
                dataset = load_dataset(path=path, split="test")
                datasets.append(dataset)
                dataset_iters.append(iter(dataset))
                datasets_name.append(path)
                weights.append(weight)

        assert len(weights) > 0, "no dataset source is chosen"
        chosen_dataset_ids = random.choices(population=range(len(weights)), weights=weights, k=num_requests)

        data_list = []
        for i in chosen_dataset_ids:
            dataset = datasets[i]
            dataset_iter = dataset_iters[i]
            name = datasets_name[i]
            data = next(dataset_iter)
            data_list.append(data)

        def create_entry(data):
            entry = SyntheticDataEntry(
                prompt = data['question'], 
                images = [encode_base64_content_from_image(data['image'])], 
            )
            return entry

        self.entries: list[SyntheticDataEntry] = []
        with ThreadPoolExecutor(max_workers=32) as executor:
            self.entries = list(executor.map(create_entry, data_list))

    def __len__(self):
        return self.num_requests

    def __getitem__(self, i: int) -> SyntheticDataEntry:
        return self.entries[i]

if __name__ == '__main__':
    import time
    start = time.perf_counter()
    dataset = SyntheticDataset(
        num_requests=10, 
        textcaps = 0, 
        pope = 1, 
        mme = 0, 
        text_vqa = 0,
        vizwiz_vqa = 0, 
    )
    end = time.perf_counter()
    print(f'dur {end - start}')
    for i in range(10):
        print(dataset[i].prompt)