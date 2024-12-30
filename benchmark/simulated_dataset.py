from PIL import Image
try:
    from dxz.engine.request import Request
except:
    from typing import Optional
    from PIL import Image
    from dataclasses import dataclass

    @dataclass
    class Request:
        prompt: str
        image: Optional[Image.Image] = None
        image_base64: str = ""
        max_tokens: int = 50

import base64

def encode_base64_content_from_path(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string

class SimulatedDataset:
    """
        an artifical dataset constructed used to profile engine in different scenarios
        all prompts' image share one image
    """
    def __init__(self, processor, image_path: str, has_images: list[bool], prompt_text_lens: list[int], output_text_lens: list[int]):
        for prompt_text_len in prompt_text_lens:
            assert prompt_text_len > 0

        for output_text_len in output_text_lens:
            assert output_text_len > 0

        self.processor = processor
        self.tokenizer = processor.tokenizer

        image = Image.open(image_path)
        self.data: list[dict] = []

        prefix = self.tokenizer.encode("USER: ")[1:]
        question = self.tokenizer.encode( "What is the content of this image?")[1:] # skip <bos>
        suffix = self.tokenizer.encode( "ASSISTANT:")[1:] # skip <bos>

        for has_image, prompt_text_len, output_text_len in zip(has_images, prompt_text_lens, output_text_lens):
            token_ids = []
            # 1. prefix
            token_ids += prefix[:prompt_text_len]
            prompt_text_len -= len(prefix)
            prompt_text_len = max(prompt_text_len, 0)
            # 2. image
            if has_image:
                token_ids += self.tokenizer.encode(processor.image_token)[1:]
            # 3. question
            suffix_len = min(len(suffix), prompt_text_len)
            prompt_text_len -= suffix_len

            token_ids += question * (prompt_text_len // len(question))
            prompt_text_len = prompt_text_len % len(question)
            token_ids += question[: prompt_text_len]
            # 4. suffix
            token_ids += suffix[:suffix_len]

            prompt = self.tokenizer.decode(token_ids, skip_special_tokens=False)
            image_base64 = encode_base64_content_from_path(image_path=image_path)
            self.data.append(Request(
                prompt = prompt, 
                image = image, 
                image_base64 = image_base64,
                max_tokens = output_text_len,
            ))
    
    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf')
    tokenizer = processor.tokenizer
    dataset = SimulatedDataset(
        processor=processor, 
        image_path='./dataset/cherry_blossom.jpg',
        has_images= [True for _ in range(0, 25)], 
        prompt_text_lens= list(range(1, 25)),
        output_text_lens= list(range(1, 25)),
        )

    for input in dataset:
        print(input['prompt'])
        print(input['max_tokens'])
        print(len(processor.tokenizer.encode(input['prompt'])))
        print('------------------------------------------------------------')
    # prompt = f"USER: <image>\nWhat is the content of this image?\nASSISTANT:"
    # token_ids = tokenizer.encode(prompt)
    # print(f"len(token_ids) {len(token_ids)}")
    # print(f"token_ids {token_ids}")
    # p = tokenizer.decode(token_ids, skip_special_tokens=True)
    # print(p)