import torch
from torch import Tensor
from typing import Optional
from transformers import AutoProcessor
from transformers import AutoTokenizer
from dxz.model.downloader import download_hf_model
from dataclasses import dataclass
from dxz.model.parameters import VisionModelParameters, VisionModelOutput, LanguageModelParameters, LanguageModelOutput

@dataclass
class VisionModelConfig:
    image_token_id: int
    num_image_tokens: int

@dataclass
class LanguageModelConfig:
    n_layers: int
    max_position_embeddings: int
    n_qo_heads: int
    n_kv_heads: int
    head_dim: int


class VisionModel:
    def forward(pixel_values, model_params: VisionModelParameters) -> VisionModelOutput:
        raise Exception('interface not implemented')

class LanguageModel:
    def forward(self, input_ids: Tensor, image_features: Optional[Tensor], position_ids: Tensor, model_params: LanguageModelParameters) -> LanguageModelOutput:
        raise Exception('interface not implemented')

class ModelFactory:
    def __init__(self, model_name: str, model_path: Optional[str] = None, dtype: torch.dtype=torch.half, device: torch.device=torch.device('cuda:0')):
        self.model_name = model_name
        if model_path is None:
            self.model_path = download_hf_model(repo_id=model_name)
        self.dtype = dtype
        self.device = device

        if model_name == "llava-hf/llava-1.5-7b-hf":
            from dxz.model.llava import LlavaLanguageModel, LlavaVisionModel
            self.vision_model_class = LlavaVisionModel
            self.language_model_class = LlavaLanguageModel
        else:
            raise Exception(f'unsupported model {model_name}')

    def getVisionModel(self) -> tuple[VisionModel, VisionModelConfig]:
        model = self.vision_model_class(self.model_path, self.dtype, self.device)
        return model, model.config

    def getLanguageModel(self) -> tuple[LanguageModel, LanguageModelConfig]:
        model = self.language_model_class(self.model_path, self.dtype, self.device)
        return model, model.config

    def getProcessor(self) -> AutoProcessor:
        return AutoProcessor.from_pretrained(self.model_path)

    def getTokenizer(self) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(self.model_path)