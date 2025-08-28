import torch
from torch import Tensor
from hydrainfer.model import getModelFactory, ModelFactoryConfig, ModelFactoryContext, VisionModel, VisionModelParameters, LanguageModel, VisionModelConfig, LanguageModelConfig
from hydrainfer.layer.causal_attention import AttentionParameters, AttentionParametersBuilder
from batch_analysis import benchmark
from dataclasses import dataclass, field
from typing import Literal
from transformers import AutoProcessor
from PIL import Image
import numpy as np

dtype = torch.half
device = torch.device('cuda:0')


@dataclass
class RequestMetaData:
    # number of n_requests same requests with a list of image with image_sizes or prompt with n_prompt_tokens or decode with n_kv_cache_tokens
    n_requests: int
    stage: list[Literal['encode', 'prefill', 'decode']]
    image_sizes: list[tuple[int, int]]
    n_prompt_tokens: int
    n_kv_cache_tokens: int


class Batch:
    def __init__(
        self, 
        vision_model: VisionModel, 
        language_model: LanguageModel, 
        vision_model_config: VisionModelConfig, 
        language_model_config: LanguageModelConfig, 
        processor: AutoProcessor, 
        requests_meta_data: list[RequestMetaData], 
    ):
        self.vision_model = vision_model
        self.language_model = language_model
        self.vision_model_config = vision_model_config
        self.language_model_config = language_model_config
        self.processor = processor
        self.requests_meta_data = requests_meta_data

        self.pixel_values: list[Tensor] = []
        for request_meta_data in requests_meta_data:
            if request_meta_data.stage != 'encode':
                continue
            for i in range(request_meta_data.n_requests):
                for w, h in range(request_meta_data.image_sizes):
                    random_array = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
                    image = Image.fromarray(random_array)
                    self.pixel_values.append()
                    images_tensor = self.processor(
                        text="", 
                        images = image, 
                        return_tensors="pt"
                    )['pixel_values']
                    self.pixel_values.append(images_tensor)

        position_ids_list: list[int] = []
        selected_token_ids_list: list[int] = []
        n_layers = self.language_model_config.n_layers
        n_tokens = 2
        block_size = 16
        n_kv_heads = self.language_model_config.n_kv_heads
        head_size = self.language_model_config.head_dim
        builder = AttentionParametersBuilder(
            num_qo_heads=self.language_model_config.n_qo_heads, 
            num_kv_heads=self.language_model_config.n_kv_heads, 
            head_dim=self.language_model_config.head_dim, 
            block_size=block_size, 
            device=device, 
        )

        total_memory = torch.cuda.get_device_properties(torch.device(self.config.model.device)).total_memory
        model_memory = torch.cuda.max_memory_allocated()
        reserved_memory * 0.8 - n_layers

        for request_meta_data in self.requests_meta_data:
            if request_meta_data.stage != 'prefill':
                continue
            for i in range(request_meta_data.n_requests):
                position_ids_list.extend(list(range(request_meta_data.n_prompt_tokens)))
                selected_token_ids_list
            



    def __call__(self):
        self.vision_model.forward(self.pixel_values, self.vision_model_parameters)


if __name__ == '__main__':
    factory = getModelFactory(ModelFactoryConfig(path='/mnt/cfs/9n-das-admin/llm_models/llava-1.5-7b-hf'), ModelFactoryContext())
    language_model = factory.getLanguageModel()
    vision_model = factory.getVisionModel()
    encode = BatchEncode(vision_model, 4)