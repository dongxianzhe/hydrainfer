import torch
from torch import Tensor
from typing import Optional
from dxz.memory.kv_cache import KVCache
from dxz.model.model_factory import VisionModel, VisionModelConfig, LanguageModel, LanguageModelConfig, ModelFactory, ModelFactoryConfig, ModelFactoryContext
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
    def __init__(self, config: ModelFactoryConfig, context: ModelFactoryContext):
        self.model_name = config.model_name
        if config.model_path is None:
            self.model_path = download_hf_model(repo_id='llava-hf/llava-1.5-7b-hf')
        else:
            self.model_path = config.model_path
        self.dtype = config.dtype
        self.device = config.device

    def getVisionModel(self) -> VisionModel:
        model = FakeVisionModel()
        return model

    def getVisionModelConfig(self) -> VisionModelConfig:
        config_ref = FakeConfig()
        config = VisionModelConfig(
            image_token_id = config_ref.image_token_id, 
            num_image_tokens = config_ref.num_image_tokens
        )
        return config

    def getLanguageModel(self) -> LanguageModel:
        model = FakeLanguageModel()
        return model

    def getLanguageModelConfig(self) -> LanguageModelConfig:
        config_ref = FakeConfig()
        config = LanguageModelConfig(
            n_layers = config_ref.n_layers, 
            max_position_embeddings = config_ref.max_position_embeddings, 
            n_qo_heads = config_ref.n_qo_heads, 
            n_kv_heads = config_ref.n_kv_heads, 
            head_dim = config_ref.head_dim, 
        )
        return config

    def getProcessor(self) -> AutoProcessor:
        return AutoProcessor.from_pretrained(self.model_path)

    def getTokenizer(self) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(self.model_path)