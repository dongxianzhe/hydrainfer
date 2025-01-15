import torch
from torch import Tensor
from typing import Optional
from dxz.memory.kv_cache import KVCache
from dxz.model.model_factory import VisionModel, VisionModelConfig, LanguageModel, LanguageModelConfig, ModelFactory
from dxz.model.parameters import VisionModelOutput, VisionModelParameters, LanguageModelOutput, LanguageModelParameters
from dxz.model.downloader import download_hf_model
from transformers import AutoProcessor, AutoTokenizer
from dataclasses import dataclass


@dataclass
class FakeConfig:
    image_token_id: int = 32000
    num_image_tokens: int = 576

    n_layers: int = 1
    max_position_embeddings: int = 4096
    hidden_size: int = 4096
    n_qo_heads: int = 32
    n_kv_heads: int = 32
    head_dim: int = 128
    vocab_size = 32000
    

class FakeVisionModel(VisionModel):
    def __init__(self):
        super().__init__()
        self.config = FakeConfig()

    def forward(self, pixel_values: Tensor, model_params: VisionModelParameters) -> VisionModelOutput:
        n_images = pixel_values.shape[0]
        return VisionModelOutput(
            image_features = torch.randn(size=(n_images, self.config.num_image_tokens, self.config.hidden_size), dtype=pixel_values.dtype, device=pixel_values.device), 
            scores = None
        )


class FakeLanguageModel(LanguageModel):
    def __init__(self):
        super().__init__()
        self.config = FakeConfig()

    def forward(self, input_ids: Tensor, image_features: Optional[Tensor], position_ids: Tensor, model_params: LanguageModelParameters) -> LanguageModelOutput:
        n_tokens = input_ids.shape[0]
        return LanguageModelOutput(sample_token_ids = torch.randint(0, self.config.vocab_size, size=(n_tokens, ), dtype=input_ids.dtype, device=input_ids.device))


class FakeModelFactory(ModelFactory):
    def __init__(self, model_name: str, model_path: Optional[str] = None, dtype: torch.dtype=torch.half, device: torch.device=torch.device('cuda:0')):
        self.model_name = model_name
        if model_path is None:
            self.model_path = download_hf_model(repo_id='llava-hf/llava-1.5-7b-hf')
        else:
            self.model_path = model_path
        self.dtype = dtype
        self.device = device

    def getVisionModel(self) -> tuple[VisionModel, VisionModelConfig]:
        model = FakeVisionModel()
        config = model.config
        return FakeVisionModel(), VisionModelConfig(
            image_token_id = config.image_token_id, 
            num_image_tokens = config.num_image_tokens
        )

    def getLanguageModel(self) -> tuple[LanguageModel, LanguageModelConfig]:
        model = FakeLanguageModel()
        config = model.config
        return model, LanguageModelConfig(
            n_layers = config.n_layers, 
            max_position_embeddings = config.max_position_embeddings, 
            n_qo_heads = config.n_qo_heads, 
            n_kv_heads = config.n_kv_heads, 
            head_dim = config.head_dim, 
        )

    def getProcessor(self) -> AutoProcessor:
        return AutoProcessor.from_pretrained(self.model_path)

    def getTokenizer(self) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(self.model_path)