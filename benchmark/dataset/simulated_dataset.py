from PIL import Image
import base64
from dataclasses import dataclass


def encode_base64_content_from_path(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string


@dataclass
class SimulatedDataEntry:
    prompt: str
    image: Image.Image 
    image_base64: str


class SimulatedDataset:
    """
        an artifical dataset constructed used to profile engine in different scenarios
        all prompts' image share one image
        and prompts are same except length
        if has image is true, SimulatedDataEntry will has an image and it's base64 encoding
        prompt_text_len is text token len, not including image
    """
    def __init__(self, tokenizer, image_path: str, has_images: list[bool], prompt_text_lens: list[int]):
        for prompt_text_len in prompt_text_lens:
            assert prompt_text_len > 0

        self.tokenizer = tokenizer

        image = Image.open(image_path)
        image_base64 = encode_base64_content_from_path(image_path=image_path)
        self.dataset: list[SimulatedDataEntry] = []

        max_tokens = max(prompt_text_lens)
        question = self.tokenizer.encode( "What is the content of this image?")[1:] # skip <bos>
        question = question * ((max_tokens  + len(question) - 1) // len(question))

        for has_image, prompt_text_len in zip(has_images, prompt_text_lens):
            if has_image:
                prompt = "<image>" + tokenizer.decode(question[:prompt_text_len])
            else:
                prompt = tokenizer.decode(question[:prompt_text_len])
            
            self.dataset.append(SimulatedDataEntry(
                prompt = prompt, 
                image = image if has_image else None, 
                image_base64 = image_base64 if has_image else None, 
            ))
    
    def __getitem__(self, index: int) -> SimulatedDataEntry:
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf')
    tokenizer = processor.tokenizer
    dataset = SimulatedDataset(
        tokenizer=tokenizer, 
        image_path='./cherry_blossom.jpg',
        has_images= [i < 12 for i in range(0, 25)], 
        prompt_text_lens= list(range(1, 25)),
        )

    for i, entry in enumerate(dataset):
        print(entry.prompt)