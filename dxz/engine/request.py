from typing import Optional
from PIL import Image
from dataclasses import dataclass

@dataclass
class Request:
    prompt: str
    image: Optional[Image.Image] = None
    image_base64: str = ""
    max_tokens: int = 50