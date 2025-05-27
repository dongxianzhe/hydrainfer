import os
import json
import torch
from torch import Tensor
from typing import Optional
from transformers import AutoProcessor
from transformers import AutoTokenizer
from dataclasses import dataclass, field, fields
import argparse
from hydrainfer.model.downloader import download_hf_model
from hydrainfer.utils.torch_utils import str2dtype, str2device
from hydrainfer.model.parameters import VisionModelParameters, VisionModelOutput, LanguageModelParameters, LanguageModelOutput
from hydrainfer.model_parallel.process_group import ParallelConfig, ProcessGroup


class ImageTokenCaculator:
    def get_num_image_tokens(self, image_size: tuple[int, int]) -> int:
        # image_size (height, width)
        raise NotImplemented


@dataclass
class VisionModelConfig:
    image_token: str # llava: "<image>", qwenvl2: "<|vision_start|><|image_pad|><|vision_end|>"
    image_token_id: int # llava: token id of "<image>", qwenvl2: token id of "<|image_pad|>"
    image_token_caculator: ImageTokenCaculator # used to determine number of image tokens per image in llavanext model which is caculated based on image resolution
    


@dataclass
class LanguageModelConfig:
    n_layers: int
    max_position_embeddings: int
    n_qo_heads: int
    n_kv_heads: int
    head_dim: int
    eos_token_id: int


class VisionModel:
    def forward(self, pixel_values: list[Tensor], model_params: VisionModelParameters) -> VisionModelOutput:
        # llava1.5 [(3, 336, 336), (3, 336, 336), (3, 336, 336) ...]
        # llavanext [(n_patches, 3, 336, 336), (n_patches, 3, 336, 336), (n_patches, 3, 336, 336) ...]
        # qwen2L [(n_patches, 1176), (n_patches, 1176), (n_patches, 1176) ...]
        raise NotImplementedError


class LanguageModel:
    def forward(self, input_ids: Tensor, image_features: Optional[Tensor], position_ids: Tensor, model_params: LanguageModelParameters) -> LanguageModelOutput:
        raise NotImplementedError


class Tokenizer:
    def encode(self, prompt: str) -> list[int]:
        raise NotImplementedError
    def decode(self, token_id: int) -> str:
        raise NotImplementedError
    def apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        """
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        """
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

    def getTokenizer(self) -> Tokenizer:
        raise NotImplementedError


@dataclass
class ModelFactoryConfig:
    path: str = "llava-hf/llava-1.5-7b-hf"
    dtype: str = "fp16"
    device: str = "cuda:0"


@dataclass
class ModelFactoryContext:
    process_group: Optional[ProcessGroup] = None


def getModelFactory(config: ModelFactoryConfig, context: ModelFactoryContext) -> ModelFactory:
    if not os.path.isdir(config.path):
        config.path = download_hf_model(repo_id=config.path)

    with open(os.path.join(config.path, "config.json"), "r", encoding="utf-8") as file:
        json_config = json.load(file)
    architecture = json_config.get("architectures", [None])[0]
    model_type = json_config.get("model_type", None)

    if architecture == 'LlavaForConditionalGeneration':
        from hydrainfer.model.llava import LlavaModelFactory
        return LlavaModelFactory(config, context)
    if architecture == 'LlavaNextForConditionalGeneration':
        from hydrainfer.model.llavanext import LlavaNextModelFactory
        return LlavaNextModelFactory(config, context)
    if architecture == 'Qwen2VLForConditionalGeneration':
        from hydrainfer.model.qwen2_vl import Qwen2VLModelFactory
        return Qwen2VLModelFactory(config, context)
    if model_type == 'deepseek_vl_v2':
        from hydrainfer.model.deepseek_vl2 import DeepSeekVL2ModelFactory
        return DeepSeekVL2ModelFactory(config, context)

    raise Exception(f'unsupported model architecture model_type {architecture} or {model_type}')