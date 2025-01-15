import torch
from torch import Tensor
from typing import Optional
from transformers import AutoProcessor
from transformers import AutoTokenizer
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
    def forward(self, pixel_values: Tensor, model_params: VisionModelParameters) -> VisionModelOutput:
        raise Exception('interface not implemented')


class LanguageModel:
    def forward(self, input_ids: Tensor, image_features: Optional[Tensor], position_ids: Tensor, model_params: LanguageModelParameters) -> LanguageModelOutput:
        raise Exception('interface not implemented')


class ModelFactory:
    def getVisionModel(self) -> tuple[VisionModel, VisionModelConfig]:
        raise Exception(f'interface not implemented')

    def getLanguageModel(self) -> tuple[LanguageModel, LanguageModelConfig]:
        raise Exception(f'interface not implemented')

    def getProcessor(self) -> AutoProcessor:
        raise Exception(f'interface not implemented')

    def getTokenizer(self) -> AutoTokenizer:
        raise Exception(f'interface not implemented')


def getModelFactory(model_name: str, model_path: Optional[str] = None, dtype: torch.dtype=torch.half, device: torch.device=torch.device('cuda:0')) -> ModelFactory:
    if model_name == "llava-hf/llava-1.5-7b-hf":
        from dxz.model.llava import LlavaModelFactory
        return LlavaModelFactory(model_name, model_path, dtype, device)
    if model_name == "gpt2":
        from dxz.model.gpt2 import GPT2ModelFactory
        return GPT2ModelFactory(model_name, model_path, dtype, device)
    if model_name == 'meta-llama/Llama-2-7b-hf':
        from dxz.model.llama import LlamaModelFactory
        return LlamaModelFactory(model_name, model_path, dtype, device)
    if model_name == "fake":
        from dxz.model.fake import FakeModelFactory
        return FakeModelFactory(model_name, model_path, dtype, device)
    raise Exception(f'invalid model {model_name}')
