import os
import io
from PIL import Image
import pandas as pd
from dataclasses import dataclass

@dataclass
class Question:
    question_id: str
    image: Image
    question: str
    reference_answer: str
    category: str
    answer: str
class MMETestPaper:
    """
        -----------------------------answer rules-------------------------------
        total question: 2374
        each image have two question
        accuracy   is the number of correct answer divided by the total question
        accuracy++ is the number of images where both questions are answered correctly divided by the total number of images
        There are a total of 14 categories, with a maximum score of 2800 points.
        Each category will have two evaluation metrics: accuracy and accuracy++. Each metric is worth 100 points, for a total of 200 points.
        total category: 14
        categorys     : existence, OCR, posters, artwork, numerical_calculation, color, landmark, text_translation, position, count, commonsense_reasoning, scene, celebrity, code_reasoning
        the total number question each category is:
            code_reasoning : 40
            artwork : 400
            celebrity : 340
            numerical_calculation : 40
            text_translation : 40
            count : 60
            color : 60
            commonsense_reasoning : 140
            position : 60
            OCR : 40
            landmark : 400
            scene : 400
            existence : 60
            posters : 294
        perception score is category score sum of 
            existence
            count
            color
            position
            posters
            celebrity
            scene
            landmark
            artwork
            OCR
        cognition score is category score sum of
            commonsense_reasoning
            numerical_calculation
            text_translation
            code_reasoning
    """
    def __init__(self, path: str=""):
        files = [
            os.path.join(path, "test-00000-of-00004-a25dbe3b44c4fda6.parquet"), 
            os.path.join(path, "test-00001-of-00004-7d22c7f1aba6fca4.parquet"), 
            os.path.join(path, "test-00002-of-00004-594798fd3f5b029c.parquet"), 
            os.path.join(path, "test-00003-of-00004-53ae1794f93b1e35.parquet"), 
        ]
        self.questions = []
        for file in files:
            df = pd.read_parquet(file)
            for _, row in df.iterrows():
                self.questions.append(Question(
                    question_id = row['question_id'],
                    image = Image.open(io.BytesIO(row['image']['bytes'])),
                    question = row['question'],
                    reference_answer = row['answer'],
                    category = row['category'], 
                    answer = "",
                ))

            self.categorys = set([question.category for question in self.questions])
            self.category_count = {}
            for question in self.questions:
                self.category_count[question.category] = self.category_count.get(question.category, 0) + 1

    def mark_paper(self):
        marks = { 
            'color_accuracy' : 0., 
            'scene_accuracy' : 0., 
            'code_reasoning_accuracy' : 0., 
            'OCR_accuracy' : 0., 
            'commonsense_reasoning_accuracy' : 0., 
            'landmark_accuracy' : 0., 
            'numerical_calculation_accuracy' : 0., 
            'position_accuracy' : 0., 
            'existence_accuracy' : 0., 
            'celebrity_accuracy' : 0., 
            'posters_accuracy' : 0., 
            'artwork_accuracy' : 0., 
            'text_translation_accuracy' : 0., 
            'count_accuracy' : 0., 

            'color_accuracy++' : 0. , 
            'scene_accuracy++' : 0. , 
            'code_reasoning_accuracy++' : 0. , 
            'OCR_accuracy++' : 0. , 
            'commonsense_reasoning_accuracy++' : 0. , 
            'landmark_accuracy++' : 0. , 
            'numerical_calculation_accuracy++' : 0. , 
            'position_accuracy++' : 0. , 
            'existence_accuracy++' : 0. , 
            'celebrity_accuracy++' : 0. , 
            'posters_accuracy++' : 0. , 
            'artwork_accuracy++' : 0. , 
            'text_translation_accuracy++' : 0. , 
            'count_accuracy++' : 0. , 

            'perception_score' : 0., 
            'cognition_score' : 0., 
        }
        for _, (question1, question2) in enumerate(zip(self.questions[::2], self.questions[1::2])):
            assert question1.question_id == question2.question_id
            category = question1.category
            mark1 = category + "_accuracy"
            mark2 = category + "_accuracy++"
            if question1.answer == question1.reference_answer:
                marks[mark1] += 1
            if question2.answer == question2.reference_answer:
                marks[mark1] += 1
            if question1.answer == question1.reference_answer and question2.answer == question2.reference_answer:
                marks[mark2] += 1
        
        marks['color_accuracy']                   = marks['color_accuracy']                   / (self.category_count['color'])                     * 100
        marks['scene_accuracy']                   = marks['scene_accuracy']                   / (self.category_count['scene'])                     * 100
        marks['code_reasoning_accuracy']          = marks['code_reasoning_accuracy']          / (self.category_count['code_reasoning'])            * 100
        marks['OCR_accuracy']                     = marks['OCR_accuracy']                     / (self.category_count['OCR'])                       * 100
        marks['commonsense_reasoning_accuracy']   = marks['commonsense_reasoning_accuracy']   / (self.category_count['commonsense_reasoning'])     * 100
        marks['landmark_accuracy']                = marks['landmark_accuracy']                / (self.category_count['landmark'])                  * 100
        marks['numerical_calculation_accuracy']   = marks['numerical_calculation_accuracy']   / (self.category_count['numerical_calculation'])     * 100
        marks['position_accuracy']                = marks['position_accuracy']                / (self.category_count['position'])                  * 100
        marks['existence_accuracy']               = marks['existence_accuracy']               / (self.category_count['existence'])                 * 100
        marks['celebrity_accuracy']               = marks['celebrity_accuracy']               / (self.category_count['celebrity'])                 * 100
        marks['posters_accuracy']                 = marks['posters_accuracy']                 / (self.category_count['posters'])                   * 100
        marks['artwork_accuracy']                 = marks['artwork_accuracy']                 / (self.category_count['artwork'])                   * 100
        marks['text_translation_accuracy']        = marks['text_translation_accuracy']        / (self.category_count['text_translation'])          * 100
        marks['count_accuracy']                   = marks['count_accuracy']                   / (self.category_count['count'])                     * 100
        marks['color_accuracy++']                 = marks['color_accuracy++']                 / (self.category_count['color'] / 2)                 * 100
        marks['scene_accuracy++']                 = marks['scene_accuracy++']                 / (self.category_count['scene'] / 2)                 * 100
        marks['code_reasoning_accuracy++']        = marks['code_reasoning_accuracy++']        / (self.category_count['code_reasoning'] / 2)        * 100
        marks['OCR_accuracy++']                   = marks['OCR_accuracy++']                   / (self.category_count['OCR'] / 2)                   * 100
        marks['commonsense_reasoning_accuracy++'] = marks['commonsense_reasoning_accuracy++'] / (self.category_count['commonsense_reasoning'] / 2) * 100
        marks['landmark_accuracy++']              = marks['landmark_accuracy++']              / (self.category_count['landmark'] / 2)              * 100
        marks['numerical_calculation_accuracy++'] = marks['numerical_calculation_accuracy++'] / (self.category_count['numerical_calculation'] / 2) * 100
        marks['position_accuracy++']              = marks['position_accuracy++']              / (self.category_count['position'] / 2)              * 100
        marks['existence_accuracy++']             = marks['existence_accuracy++']             / (self.category_count['existence'] / 2)             * 100
        marks['celebrity_accuracy++']             = marks['celebrity_accuracy++']             / (self.category_count['celebrity'] / 2)             * 100
        marks['posters_accuracy++']               = marks['posters_accuracy++']               / (self.category_count['posters'] / 2)               * 100
        marks['artwork_accuracy++']               = marks['artwork_accuracy++']               / (self.category_count['artwork'] / 2)               * 100
        marks['text_translation_accuracy++']      = marks['text_translation_accuracy++']      / (self.category_count['text_translation'] / 2)      * 100
        marks['count_accuracy++']                 = marks['count_accuracy++']                 / (self.category_count['count'] / 2)                 * 100
        marks['perception_score'] = sum([
            marks['existence_accuracy'], 
            marks['count_accuracy'], 
            marks['color_accuracy'], 
            marks['position_accuracy'], 
            marks['posters_accuracy'], 
            marks['celebrity_accuracy'], 
            marks['scene_accuracy'], 
            marks['landmark_accuracy'], 
            marks['artwork_accuracy'], 
            marks['OCR_accuracy'], 
            marks['existence_accuracy++'], 
            marks['count_accuracy++'], 
            marks['color_accuracy++'], 
            marks['position_accuracy++'], 
            marks['posters_accuracy++'], 
            marks['celebrity_accuracy++'], 
            marks['scene_accuracy++'], 
            marks['landmark_accuracy++'], 
            marks['artwork_accuracy++'], 
            marks['OCR_accuracy++'],
        ])
        marks['cognition_score'] = sum([
            marks['commonsense_reasoning_accuracy'], 
            marks['numerical_calculation_accuracy'], 
            marks['text_translation_accuracy'], 
            marks['code_reasoning_accuracy'], 
            marks['commonsense_reasoning_accuracy++'], 
            marks['numerical_calculation_accuracy++'], 
            marks['text_translation_accuracy++'], 
            marks['code_reasoning_accuracy++'], 
        ])
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
    paper = MMETestPaper()
    for question in tqdm(paper.questions):
        inputs = processor(
            text=f"USER: <image>\n{question.question}?\nAnswer the question using a single word or phrase. ASSISTANT:", 
            images=question.image,
            return_tensors="pt"
        )
        for key in inputs.keys():
            inputs[key] = inputs[key].to(device)
        generated_ids = model.generate(**inputs, max_new_tokens=15)

        response = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if "Yes" in response[0]:
            question.answer = "Yes"
        elif "No" in response[0]:
            question.answer = "No"
        else:
            print('unclear')
    marks = paper.mark_paper()
    print(marks)

def test_engine():
    from tqdm import tqdm
    import torch
    from dxz.engine.engine import EngineConfig, Engine, SchedulerConfig
    from dxz.memory.virtual_kv_cache import MemoryConfig
    from dxz.memory.request_processor import RequestProcessorConfig
    paper = MMETestPaper()
    config = EngineConfig(
        model_name = "llava-hf/llava-1.5-7b-hf", 
        dtype = torch.half, 
        device = torch.device('cuda:0'), 
        memory_config=MemoryConfig(
            num_blocks = 20000, 
            block_size = 16, 
        ), 
        scheduler_config=SchedulerConfig(
            batch_policy = 'continuousbatch', 
            max_running_requests = 10, 
        ), 
        request_processor_config=RequestProcessorConfig(
            max_tokens = 64, 
            kv_cache_eviction_policy = None, 
            window_size = 28, 
            attention_sink_size = 4, 
            token_pruning_policy = None, 
            n_embed_output_tokens = 256, 
        ), 
    )
    engine = Engine(config)
    inputs = [{
        "prompt":f"USER: <image>\n{question.question}?\nAnswer the question using a single word or phrase. ASSISTANT:", 
        "multi_modal_data":{"image":question.image}, 
        "max_tokens":16, 
        "eos_token_id":2} for question in paper.questions]

    outputs = engine.generate(inputs)

    for i, question in enumerate(paper.questions):
        output_text = outputs[i].text
        print(output_text)
        if "Yes" in output_text:
            question.answer = "Yes"
        elif "No" in output_text:
            question.answer = "No"
        else:
            print('unclear')
    marks = paper.mark_paper()
    print(marks)

if __name__ == '__main__':
    # test_transformers()
    test_engine()