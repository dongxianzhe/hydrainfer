import io
from PIL import Image
import pandas as pd

class MMETestPaper:
    def __init__(self):
        files = [
            "test-00000-of-00004-a25dbe3b44c4fda6.parquet", 
            "test-00001-of-00004-7d22c7f1aba6fca4.parquet", 
            "test-00002-of-00004-594798fd3f5b029c.parquet", 
            "test-00003-of-00004-53ae1794f93b1e35.parquet", 
        ]
        self.questions = []
        for file in files:
            df = pd.read_parquet(file)
            for _, row in df.iterrows():
                question = {
                    "question_id": row['question_id'],
                    "image": Image.open(io.BytesIO(row['image']['bytes'])),
                    "question": row['question'],
                    "reference_answer": row['answer'],
                    "category": row['category'], 
                    "answer" : "",
                }
                self.questions.append(question)

            self.categorys = set([question['category'] for question in self.questions])
            self.category_count = {}
            for question in self.questions:
                self.category_count[question['category']] = self.category_count.get(question['category'], 0) + 1
    
    def answer_rules(self):
        print('-----------------------------answer rules-------------------------------')
        print(f'total question: {len(self.questions)}')
        print(f'each image have two question')
        print(f'accuracy   is the number of correct answer divided by the total question')
        print(f'accuracy++ is the number of images where both questions are answered correctly divided by the total number of images')
        print(f'There are a total of 14 categories, with a maximum score of 2800 points.')
        print(f'Each category will have two evaluation metrics: accuracy and accuracy++. Each metric is worth 100 points, for a total of 200 points.')
        print(f'total category: 14')
        print(f'categorys     : existence, OCR, posters, artwork, numerical_calculation, color, landmark, text_translation, position, count, commonsense_reasoning, scene, celebrity, code_reasoning')
        print(f'the total number question each category is:')
        print(f'    code_reasoning : 40')
        print(f'    artwork : 400')
        print(f'    celebrity : 340')
        print(f'    numerical_calculation : 40')
        print(f'    text_translation : 40')
        print(f'    count : 60')
        print(f'    color : 60')
        print(f'    commonsense_reasoning : 140')
        print(f'    position : 60')
        print(f'    OCR : 40')
        print(f'    landmark : 400')
        print(f'    scene : 400')
        print(f'    existence : 60')
        print(f'    posters : 294')
        print(f'perception score is category score sum of ')
        print(f'    existence')
        print(f'    count')
        print(f'    color')
        print(f'    position')
        print(f'    posters')
        print(f'    celebrity')
        print(f'    scene')
        print(f'    landmark')
        print(f'    artwork')
        print(f'    OCR')
        print(f'cognition score is category score sum of')
        print(f'    commonsense_reasoning')
        print(f'    numerical_calculation')
        print(f'    text_translation')
        print(f'    code_reasoning')

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
            assert question1['question_id'] == question2['question_id']
            category = question1['category']
            mark1 = category + "_accuracy"
            mark2 = category + "_accuracy++"
            if question1['answer'] == question1['reference_answer']:
                marks[mark1] += 1
            if question2['answer'] == question2['reference_answer']:
                marks[mark1] += 1
            if question1['answer'] == question1['reference_answer'] and question2['answer'] == question2['reference_answer']:
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
    inputs = processor(
        text="USER: <image>\nis this a panda?\nAnswer the question using a single word or phrase. ASSISTANT:", 
        images=Image.open("/home/xzd/projects/dxz/learn/llava/panda.png"), 
        return_tensors="pt"
    )

    paper = MMETestPaper()
    for question in tqdm(paper.questions):
        inputs = processor(
            text=f"USER: <image>\n{question['question']}?\nAnswer the question using a single word or phrase. ASSISTANT:", 
            images=question['image'],
            return_tensors="pt"
        )
        for key in inputs.keys():
            inputs[key] = inputs[key].to(device)
        generated_ids = model.generate(**inputs, max_new_tokens=15)

        response = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if "Yes" in response[0]:
            question['answer'] = "Yes"
        elif "No" in response[0]:
            question['answer'] = "No"
        else:
            print('unclear')
    marks = paper.mark_paper()
    print(marks)

def test_engine():
    from tqdm import tqdm
    from dxz.engine.engine import Engine, EngineConfig
    paper = MMETestPaper()
    engine = Engine(EngineConfig(
        token_prunning_policy='block_prefill',
        window_size=512,
        attention_sink_size=4,
    ))
    for question in tqdm(paper.questions):
        inputs = [{"prompt":f"USER: <image>\n{question['question']}?\nAnswer the question using a single word or phrase. ASSISTANT:", "multi_modal_data":{"image":question['image']}, "max_tokens":16}]
        output_text = engine.generate(inputs)[0]
        print(output_text)
        if "Yes" in output_text:
            question['answer'] = "Yes"
        elif "No" in output_text:
            question['answer'] = "No"
        else:
            print('unclear')
    marks = paper.mark_paper()
    print(marks)

if __name__ == '__main__':
    test_engine()