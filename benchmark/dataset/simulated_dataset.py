import base64
from PIL import Image
from io import BytesIO
from dataclasses import dataclass, field, fields


def encode_base64_content_from_path(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string


def encode_base64_content_from_image(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return encoded_string

@dataclass
class SimulatedDataEntry:
    prompt: str
    images: list[Image.Image] # PIL Image used in offline inference
    image_base64: list[str] # str Image used in online inference
    n_prompt_tokens: int # prompt token numbers
    n_output_tokens: int # desired output token numbers
    ttft_slo: float
    tpot_slo: float


class SimulatedDataset:
    """
        an synthetic dataset constructed used to profile engine in different scenarios
        all prompts' image share one image
        and prompts are same except length
    """
    def __init__(self, tokenizer, image_path: str, template_prompt: str="What is the content of this image?"):
        self.tokenizer = tokenizer
        self.template_prompt_tokens = self.tokenizer.encode(template_prompt)[1:] # skip <bos>
        self.image_token = tokenizer.encode('<image>')[1:] # skip <bos>
        self.image = Image.open(image_path)
        self.image_base64 = encode_base64_content_from_path(image_path=image_path)
        self.dataset: list[SimulatedDataEntry] = []

    def append(self, n_prompt_tokens: int, n_images: int, n_output_tokens: int, ttft_slo: float, tpot_slo: float):
        assert n_prompt_tokens > 0, 'n prompt tokens should greater than 0'
        assert n_images >= 0, 'image number should greater than or equal to 0'
        assert n_output_tokens > 0, 'n output tokens should greater than 0'
        assert n_prompt_tokens >= n_images, 'prompt tokens number should at least greater than image numbers'
        self.dataset.append(self.make_data_entry(
            n_prompt_tokens = n_prompt_tokens, 
            n_images        = n_images, 
            n_output_tokens = n_output_tokens, 
            ttft_slo        = ttft_slo, 
            tpot_slo        = tpot_slo, 
        ))

    def make_data_entry(self, n_prompt_tokens: int, n_images: int, n_output_tokens: int, ttft_slo: float, tpot_slo: float) -> SimulatedDataEntry:
        prompt_tokens = self.template_prompt_tokens * ((n_prompt_tokens  + len(self.template_prompt_tokens) - 1) // len(self.template_prompt_tokens))
        prompt_tokens = self.image_token * n_images + prompt_tokens
        prompt_tokens = prompt_tokens[:n_prompt_tokens]
        prompt = self.tokenizer.decode(prompt_tokens)
        return SimulatedDataEntry(
            prompt = prompt, 
            images = [self.image] * n_images, 
            image_base64 = [self.image_base64] * n_images, 
            n_prompt_tokens = n_prompt_tokens, 
            n_output_tokens = n_output_tokens, 
            ttft_slo = ttft_slo, 
            tpot_slo = tpot_slo
        )

    def __getitem__(self, index: int) -> SimulatedDataEntry:
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class SyntheticDataset:
    def __init__(self):
        self.dataset: list[SimulatedDataEntry] = []
        pass

    def append(self, entry: SimulatedDataEntry):
        self.dataset.append(entry)

    def __getitem__(self, index: int) -> SimulatedDataEntry:
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)