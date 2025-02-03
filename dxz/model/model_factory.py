import torch
from torch import Tensor
from typing import Optional
from transformers import AutoProcessor
from transformers import AutoTokenizer
from dataclasses import dataclass, field, fields
import argparse
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


def get_device(device_type: str) -> torch.device:
    if device_type == 'cuda':
        return torch.device('cuda:0')
    if device_type == 'cpu':
        return torch.device('cpu')
    raise Exception(f'device_type {device_type} is not supported')


def get_dtype(dtype: str) -> torch.dtype:
    if dtype == 'fp16':
        return torch.half
    if dtype == 'fp32':
        return torch.float
    raise Exception(f'dtype {dtype} is not supported')


@dataclass
class ModelFactoryConfig:
    model_name: str = "llava-hf/llava-1.5-7b-hf"
    model_path: Optional[str] = None
    dtype: torch.dtype=torch.half
    device: torch.device=torch.device('cuda:0')

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> 'ModelFactoryConfig':
        attrs = [attr.name for attr in fields(cls) if attr.name not in ['dtype', 'device']]
        dtype = get_dtype(args.dtype)
        device = get_device(args.device)
        config = cls(
            device = device, 
            dtype = dtype, 
            **{attr: getattr(args, attr) for attr in attrs}
        )
        return config

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--model-name', type=str, default="llava-hf/llava-1.5-7b-hf", help='The name of the model.')
        parser.add_argument('--model_path', type=str, nargs="?", default=None, help="path to the model, if set none will download model from huggingface to default cache directory of transformers library with the model-name arg.")
        parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda', help="Specify device type: 'cuda' or 'cpu'")
        parser.add_argument('--dtype', type=str, choices=['fp16', 'fp32'], default='fp16', help="Specify data type: 'fp16' or 'fp32'")
        return parser


@dataclass
class ModelFactoryContext:
    process_group: Optional[ProcessGroup] = None


def getModelFactory(config: ModelFactoryConfig, context: ModelFactoryContext) -> ModelFactory:
    if config.model_name == "llava-hf/llava-1.5-7b-hf":
        from dxz.model.llava import LlavaModelFactory
        return LlavaModelFactory(config, context)
    if config.model_name == "gpt2":
        from dxz.model.gpt2 import GPT2ModelFactory
        return GPT2ModelFactory(config, context)
    if config.model_name == 'meta-llama/Llama-2-7b-hf':
        from dxz.model.llama import LlamaModelFactory
        return LlamaModelFactory(config, context)
    if config.model_name == "fake":
        from dxz.model.fake import FakeModelFactory
        return FakeModelFactory(config, context)
    raise Exception(f'invalid model {config.model_name}')