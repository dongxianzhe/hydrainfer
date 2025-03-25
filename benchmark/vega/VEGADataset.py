import base64
import json
import os
import io
from PIL import Image
from dataclasses import dataclass

def encode_base64_content_from_path(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string

@dataclass
class Question:
    id: str
    title: str
    context: str
    truth_fig_idx: int
    image_paths: list[str]
    caption: str
    answer: str
    question: str
    image: Image = None
    image_base64: str = None

class VEGADataset:
    def __init__(self, path: str=""):
        self.base_path = path
        files = [
            os.path.join(path, "IITC_4k_test.json"), 
        ]
        self.questions = []
        for file in files:
            with open(file, 'r', encoding='utf-8') as file_handle:
                for line in file_handle:
                    if line.strip():
                        entry = Question(**json.loads(line))
                        image_path = os.path.join(self.base_path, 'imgs', entry.image_paths[entry.truth_fig_idx])
                        entry.image_base64 = encode_base64_content_from_path(image_path)
                        self.questions.append(entry)

    def __getitem__(self, index: int) -> Question:
        return self.questions[index]

    def __len__(self):
        return len(self.questions)