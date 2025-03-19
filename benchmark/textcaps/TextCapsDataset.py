import base64
import os
import io
from PIL import Image
import pandas as pd
from dataclasses import dataclass

@dataclass
class Question:
    question_id: str
    image: Image
    image_base64: str
    question: str

class TextCapsDataset:
    def __init__(self, path: str=""):
        files = [
            os.path.join(path, "test-00001-of-00002.parquet"), 
            os.path.join(path, "test-00000-of-00002.parquet"), 
        ]
        self.questions = []
        for file in files:
            df = pd.read_parquet(file)
            for _, row in df.iterrows():
                self.questions.append(Question(
                    question_id = row['question_id'],
                    image = Image.open(io.BytesIO(row['image']['bytes'])),
                    image_base64 = base64.b64encode(row['image']['bytes']).decode('utf-8'),
                    question = row['question'],
                ))

    def __getitem__(self, index: int) -> Question:
        return self.questions[index]

    def __len__(self):
        return len(self.questions)