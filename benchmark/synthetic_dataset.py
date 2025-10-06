import os
import base64
import random
from data_preprocess import load_processed_dataset
from dataclasses import dataclass
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Union


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
    images: Optional[list[str]]
    max_tokens: int
    images_size: list[Union[tuple[int, int], list[int]]] # (width, height) we need union because dump json wil convert tuple to list but load json can't convert list to tuple
    dataset: str


class SyntheticDataset:
    def __init__(
        self, 
        model_path: str, 
        num_requests: int,
        textcaps: int = 0, 
        pope: int = 0, 
        mme: int = 0, 
        text_vqa: int = 0, 
        vizwiz_vqa: int = 0, 
    ):
        self.model_path = model_path
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
                dataset = load_processed_dataset(path=path, split="test", model_path=self.model_path)
                datasets.append(dataset)
                dataset_iters.append(iter(dataset))
                datasets_name.append(path)
                weights.append(weight)

        assert len(weights) > 0, "no dataset source is chosen"
        chosen_dataset_ids = random.choices(population=range(len(weights)), weights=weights, k=num_requests)

        tasks = []
        for i in chosen_dataset_ids:
            dataset = datasets[i]
            name = datasets_name[i]

            dataset_iter = dataset_iters[i]
            try:
                data = next(dataset_iter)
            except StopIteration:
                dataset_iters[i] = iter(dataset)
                dataset_iter = dataset_iters[i]
                data = next(dataset_iter)
            tasks.append((name, data))

        def create_entry(task):
            name, data = task
            entry = SyntheticDataEntry(
                prompt = data['question'], 
                images = [encode_base64_content_from_image(data['image'])], 
                max_tokens = data['max_tokens'], 
                images_size = [data['image'].size], 
                dataset = name, 
            )
            return entry

        self.entries: list[SyntheticDataEntry] = []
        with ThreadPoolExecutor(max_workers=32) as executor:
            self.entries = list(executor.map(create_entry, tasks))

    def __len__(self):
        return self.num_requests

    def __getitem__(self, i: int) -> SyntheticDataEntry:
        return self.entries[i]

if __name__ == '__main__':
    import time
    start = time.perf_counter()
    dataset = SyntheticDataset(
        model_path='/mnt/cfs/9n-das-admin/llm_models/llava-v1.6-vicuna-7b-hf', 
        num_requests=3000, 
        textcaps = 0, 
        pope = 0, 
        mme = 1, 
        text_vqa = 0,
        vizwiz_vqa = 0, 
    )
    end = time.perf_counter()
    print(f'dur {end - start}')
    print(len(dataset))
    for i in range(3):
        print(dataset[i].prompt)
        print(dataset[i].images_size)
        print(dataset[i].dataset)