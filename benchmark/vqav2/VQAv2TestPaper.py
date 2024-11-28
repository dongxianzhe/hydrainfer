import io
import pandas as pd
from PIL import Image
from dataclasses import dataclass

@dataclass
class Question:
    """
    https://huggingface.co/datasets/lmms-lab/VQAv2/viewer/default/validation
    we only get a subset columns in the dataset
    """
    reference_answers: list[str]
    question_id: int
    question: str
    image: Image
    answer: str

class VQAv2TestPaper:
    def __init__(self):
        val_files = [
            "validation-00000-of-00068.parquet", 
            "validation-00001-of-00068.parquet", 
            "validation-00002-of-00068.parquet", 
            "validation-00003-of-00068.parquet", 
            "validation-00004-of-00068.parquet", 
            "validation-00005-of-00068.parquet", 
            "validation-00006-of-00068.parquet", 
            "validation-00007-of-00068.parquet", 
            "validation-00008-of-00068.parquet", 
            "validation-00009-of-00068.parquet", 
            "validation-00010-of-00068.parquet", 
            "validation-00011-of-00068.parquet", 
            "validation-00012-of-00068.parquet", 
            "validation-00013-of-00068.parquet",
            "validation-00014-of-00068.parquet",
            "validation-00015-of-00068.parquet",
            "validation-00016-of-00068.parquet",
            "validation-00017-of-00068.parquet",
            "validation-00018-of-00068.parquet",
            "validation-00019-of-00068.parquet",
            "validation-00020-of-00068.parquet",
            "validation-00021-of-00068.parquet",
            "validation-00022-of-00068.parquet",
            "validation-00023-of-00068.parquet",
            "validation-00024-of-00068.parquet",
            "validation-00025-of-00068.parquet",
            "validation-00026-of-00068.parquet",
            "validation-00027-of-00068.parquet",
            "validation-00028-of-00068.parquet",
            "validation-00029-of-00068.parquet",
            "validation-00030-of-00068.parquet",
            "validation-00031-of-00068.parquet",
            "validation-00032-of-00068.parquet",
            "validation-00033-of-00068.parquet",
            "validation-00034-of-00068.parquet",
            "validation-00035-of-00068.parquet",
            "validation-00036-of-00068.parquet",
            "validation-00037-of-00068.parquet",
            "validation-00038-of-00068.parquet",
            "validation-00039-of-00068.parquet",
            "validation-00040-of-00068.parquet",
            "validation-00041-of-00068.parquet",
            "validation-00042-of-00068.parquet",
            "validation-00043-of-00068.parquet",
            "validation-00044-of-00068.parquet",
            "validation-00045-of-00068.parquet",
            "validation-00046-of-00068.parquet",
            "validation-00047-of-00068.parquet",
            "validation-00048-of-00068.parquet",
            "validation-00049-of-00068.parquet",
            "validation-00050-of-00068.parquet",
            "validation-00051-of-00068.parquet",
            "validation-00052-of-00068.parquet",
            "validation-00053-of-00068.parquet",
            "validation-00054-of-00068.parquet",
            "validation-00055-of-00068.parquet",
            "validation-00056-of-00068.parquet",
            "validation-00057-of-00068.parquet",
            "validation-00058-of-00068.parquet",
            "validation-00059-of-00068.parquet",
            "validation-00060-of-00068.parquet",
            "validation-00061-of-00068.parquet",
            "validation-00062-of-00068.parquet",
            "validation-00063-of-00068.parquet",
            "validation-00064-of-00068.parquet",
            "validation-00065-of-00068.parquet",
            "validation-00066-of-00068.parquet",
            "validation-00067-of-00068.parquet",
        ]
        df = pd.read_parquet(val_files)
        self.questions: list[Question] = []
        for _, row in df.iterrows():
            self.questions.append(Question(
                question_id         = row["question_id"], 
                question            = row["question"], 
                image               = Image.open(io.BytesIO(row['image']['bytes'])), 
                reference_answers   = list(t['answer'] for t in row["answers"]), 
                answer              = ""
            ))

        assert len(self.questions) == 214354, f'{len(self.questions)}'
    
    def mark_paper(self):
        marks = {
            "accuracy":0,
        }
        for question in self.questions:
            marks['accuracy'] += min(1, question.reference_answers.count(question.answer) / 3)
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
    paper = VQAv2TestPaper()
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