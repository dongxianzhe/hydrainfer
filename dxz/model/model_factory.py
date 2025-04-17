import torch
from torch import Tensor
from typing import Optional
from transformers import AutoProcessor
from transformers import AutoTokenizer
from dataclasses import dataclass, field, fields
import argparse
from dxz.utils.torch_utils import str2dtype, str2device
from dxz.model.parameters import VisionModelParameters, VisionModelOutput, LanguageModelParameters, LanguageModelOutput
from dxz.model_parallel.process_group import ParallelConfig, ProcessGroup

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
        raise NotImplementedError


class LanguageModel:
    def forward(self, input_ids: Tensor, image_features: Optional[Tensor], position_ids: Tensor, model_params: LanguageModelParameters) -> LanguageModelOutput:
        raise NotImplementedError


class ModelFactory:
    def getVisionModel(self) -> VisionModel:
        raise NotImplementedError

    def getLanguageModel(self) -> LanguageModel:
        raise NotImplementedError

    def getVisionModelConfig(self) -> VisionModelConfig: 
        raise NotImplementedError

    def getLanguageModelConfig(self) -> LanguageModelConfig: 
        raise NotImplementedError

    def getProcessor(self) -> AutoProcessor:
        raise NotImplementedError

    def getTokenizer(self) -> AutoTokenizer:
        raise NotImplementedError


@dataclass
class ModelFactoryConfig:
    name: str = "llava-hf/llava-1.5-7b-hf"
    path: Optional[str] = None
    dtype: str = "fp16"
    device: str = "cuda:0"


@dataclass
class ModelFactoryContext:
    process_group: Optional[ProcessGroup] = None


def getModelFactory(config: ModelFactoryConfig, context: ModelFactoryContext) -> ModelFactory:
    if config.name == "llava-hf/llava-1.5-7b-hf":
        from dxz.model.llava import LlavaModelFactory
        return LlavaModelFactory(config, context)
    if config.name == "gpt2":
        from dxz.model.gpt2 import GPT2ModelFactory
        return GPT2ModelFactory(config, context)
    if config.name == 'meta-llama/Llama-2-7b-hf':
        from dxz.model.llama import LlamaModelFactory
        return LlamaModelFactory(config, context)
    if config.name == "fake":
        from dxz.model.fake import FakeModelFactory
        return FakeModelFactory(config, context)
    raise Exception(f'invalid model {config.name}')