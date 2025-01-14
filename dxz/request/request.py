from typing import Optional
from PIL import Image
from dataclasses import dataclass, field

@dataclass
class SamplingParameters:
    max_tokens: int = 50

@dataclass
class Request:
    request_id: int = 0
    prompt: str = ""
    image: Optional[Image.Image] = None
    image_base64: str = ""
    sampling_params: SamplingParameters = field(default_factory=SamplingParameters)