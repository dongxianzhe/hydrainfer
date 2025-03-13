import os
import io
from PIL import Image
import pandas as pd
from dataclasses import dataclass

@dataclass
class Question:
    id: str
    question_id: str
    question: str
    reference_answer: str
    image: Image
    answer: str
    
class POPETestPaper:
    """
    https://huggingface.co/datasets/lmms-lab/POPE/viewer/default/test
    9000 true or false questions
    """
    def __init__(self, path: str=""):
        files = [
            os.path.join(path, "test-00000-of-00003.parquet"), 
            os.path.join(path, "test-00001-of-00003.parquet"), 
            os.path.join(path, "test-00002-of-00003.parquet"), 
        ]
        self.questions: list[Question] = []
        for file in files:
            df = pd.read_parquet(file)
            for _, row in df.iterrows():
                self.questions.append(Question(
                    id = row['id'], 
                    question_id = row['question_id'], 
                    image = Image.open(io.BytesIO(row['image']['bytes'])), 
                    question = row['question'], 
                    reference_answer = row['answer'], 
                    answer = "", 
                ))
        assert len(self.questions) == 9000

    def mark_paper(self):
        marks = {
            "accuracy" : 0
        }
        for question in self.questions:
            if question.answer.lower() == question.reference_answer:
                marks['accuracy'] += 1
        marks['accuracy'] /= len(self.questions)
        return marks

def test_transformers():
    from tqdm import tqdm
    import torch
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    device = torch.device('cuda:0')
    model_name_or_path = "llava-hf/llava-1.5-7b-hf"
    model = LlavaForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=model_name_or_path, device_map=device, torch_dtype=torch.half
    )
    processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path=model_name_or_path)
    paper = POPETestPaper()
    for question in tqdm(paper.questions):
        text = f"USER: <image>\n{question.question}?\nWhen the provided information is insufficient, respond with 'Unanswerable'.\nAnswer the question using a single word or phrase. ASSISTANT:"
        inputs = processor(
            text=text, 
            images=question.image,
            return_tensors="pt"
        )
        for key in inputs.keys():
            inputs[key] = inputs[key].to(device)
        generated_ids = model.generate(**inputs, max_new_tokens=15)

        question.answer = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].split('ASSISTANT:')[1].strip()

    marks = paper.mark_paper()
    print(marks)

if __name__ == '__main__':
    test_transformers()