from PIL.Image import Image
from torch import Tensor
from hydrainfer.model.model_factory import ImageProcessor
from transformers import AutoProcessor

class TransformersAutoProcessorAdapter(ImageProcessor):
    def __init__(self, path: str) -> None:
        self.processor = AutoProcessor.from_pretrained(path)

    def process(self, image: Image) -> Tensor:
        images_tensor: Tensor = self.processor(
            text="", 
            images = image, 
            return_tensors="pt"
        )['pixel_values']
        return images_tensor