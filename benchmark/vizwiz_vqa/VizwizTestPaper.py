import io
from PIL import Image
import pandas as pd
from dataclasses import dataclass

@dataclass
class Question:
    question_id: str
    image: Image
    question: str
    reference_answers: list[str]
    category: str
    answer: str

class VizwizTestPaper:
    """
    test_questions has 8000 question
    val_questions has 4319 question
    test_questions has no reference_answers use val_questions
    answer the question in question.answer attribute
    """
    def __init__(self):
        test_files = [
            # "test-00000-of-00008-8bb04d0d8c47d4ab.parquet", 
            # "test-00001-of-00008-0deaf01822797f02.parquet", 
            # "test-00002-of-00008-a4468dc23224a358.parquet", 
            # "test-00003-of-00008-e3b828493e3460e9.parquet", 
            # "test-00004-of-00008-6e3fe278e8b97bab.parquet", 
            # "test-00005-of-00008-aee7903216b03f1e.parquet", 
            # "test-00006-of-00008-abcb74e67e207eb1.parquet", 
            # "test-00007-of-00008-c2a2bfd267556d6d.parquet", 
        ]
        val_files = [
            "val-00000-of-00005-7775fd61bc6a3d98.parquet", 
            "val-00001-of-00005-18e4fb673cfdd7cb.parquet", 
            "val-00002-of-00005-eeb6c831a97fe54e.parquet", 
            "val-00003-of-00005-ff7829371634e9f2.parquet", 
            "val-00004-of-00005-59be8a2af5f336e2.parquet", 
        ]

        datasets = []
        for files in [test_files, val_files]:
            questions: list[Question] = []
            for file in files:
                df = pd.read_parquet(file)
                for _, row in df.iterrows():
                    questions.append(Question(
                        question_id = row['question_id'], 
                        image = Image.open(io.BytesIO(row['image']['bytes'])), 
                        question = row['question'], 
                        reference_answers = list(row['answers']), 
                        category = row['category'], 
                        answer = "", 
                    ))
            datasets.append(questions)
        self.test_questions: list[Question]
        self.val_questions: list[Question]
        self.test_questions, self.val_questions = datasets
        # self.val_questions = self.val_questions[:10]

    def mark_paper(self):
        marks = {
            "score": 0, 
        }
        for question in self.val_questions:
            right = False
            for reference_answer in question.reference_answers:
                if reference_answer in question.answer.lower():
                    marks['score'] += 1
            if not right:
                print(f'{question.question_id} wrong: {question.reference_answers} {question.answer}')
        marks["score"] /= len(self.val_questions) * 10
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
    paper = VizwizTestPaper()
    for question in tqdm(paper.val_questions):
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