import base64
import random
from dataclasses import dataclass
from PIL import Image
from io import BytesIO
from typing import Optional, Union
from utils import load_json


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
    n_output_tokens: int
    images_size: list[Union[tuple[int, int], list[int]]] # (width, height) we need union because dump json wil convert tuple to list but load json can't convert list to tuple
    dataset: str
    n_input_tokens_ref: int = 0
    output_ref: str = ""


@dataclass
class SyntheticSourceDataset:
    model: str
    dataset: str
    model_path: str
    dataset_path: str
    entries: list[SyntheticDataEntry]


class SyntheticDataset:
    def __init__(self, num_requests: int, dataset_path_to_weight: dict[str, int]):
        from data_preprocess import get_preprocessed_data_path
        self.num_requests = num_requests
        source_datasets: list[SyntheticSourceDataset] = [load_json(data_class=SyntheticSourceDataset, data_path=get_preprocessed_data_path(dataset_path)) for dataset_path, weight in dataset_path_to_weight.items() if weight > 0]
        weights: list[int] = [weight for weight in dataset_path_to_weight.values() if weight > 0]
        assert len(weights) > 0, "no dataset source is chosen"

        self.entries: list[SyntheticDataEntry] = []
        chosen_dataset_ids = random.choices(population=range(len(source_datasets)), weights=weights, k=num_requests)
        for dataset_id, source_dataset in enumerate(source_datasets):
            n_samples = sum([chosen_dataset_id == dataset_id for chosen_dataset_id in chosen_dataset_ids])
            self.entries.extend(source_dataset.entries * (n_samples // len(source_dataset.entries)) + source_dataset.entries[:n_samples % len(source_dataset.entries)])

    def __len__(self):
        return self.num_requests

    def __getitem__(self, i: int) -> SyntheticDataEntry:
        return self.entries[i]