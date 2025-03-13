import json
import os
import io
from PIL import Image
from dataclasses import dataclass

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

class VEGADataset:
    def __init__(self, path: str=""):
        files = [
            os.path.join(path, "IITC_4k_test.json"), 
        ]
        self.questions = []
        for file in files:
            with open(file, 'r', encoding='utf-8') as file_handle:
                for line in file_handle:
                    if line.strip():
                        self.questions.append(Question(**json.loads(line)))

if __name__ == '__main__':
    dataset = VEGADataset()