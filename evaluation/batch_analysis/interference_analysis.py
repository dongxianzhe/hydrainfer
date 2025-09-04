import torch
from torch import Tensor
from hydrainfer.model import getModelFactory, ModelFactoryConfig, ModelFactoryContext, VisionModel, VisionModelParameters, LanguageModel, VisionModelConfig, LanguageModelConfig
from hydrainfer.layer.causal_attention import AttentionParameters, AttentionParametersBuilder
from hydrainfer.model import ModelProfiler, LanguageModelParameters
from hydrainfer.model import model_factory
from hydrainfer.utils.torch_utils import get_dtype_size
from hydrainfer.memory import KVCache
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
    stage: Literal['encode', 'prefill', 'decode']
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
        model_profiler: ModelProfiler, 
        processor: AutoProcessor, 
        requests_meta_data: list[RequestMetaData], 
    ):
        self.vision_model = vision_model
        self.language_model = language_model
        self.vision_model_config = vision_model_config
        self.language_model_config = language_model_config
        self.processor = processor
        self.model_profiler = model_profiler
        self.requests_meta_data = requests_meta_data

        self.has_vision_model_task = False
        original_image_sizes = []
        self.pixel_values: list[Tensor] = []
        for request_meta_data in requests_meta_data:
            if request_meta_data.stage != 'encode':
                continue

            self.has_vision_model_task = True
            for i in range(request_meta_data.n_requests):
                for w, h in request_meta_data.image_sizes:
                    random_array = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
                    image = Image.fromarray(random_array)
                    images_tensor = self.processor(
                        text="", 
                        images = image, 
                        return_tensors="pt"
                    )['pixel_values']
                    images_tensor = images_tensor.to(device)
                    self.pixel_values.append(images_tensor)
                    original_image_sizes.append((h, w))
        self.vision_model_params = VisionModelParameters(
            return_last_layer_attention = False, 
            original_image_sizes = original_image_sizes
        )

        input_ids_list: list[int] = []
        position_ids_list: list[int] = []
        selected_token_ids_list: list[int] = []

        n_layers = self.language_model_config.n_layers
        n_tokens = 2
        block_size = 16
        n_kv_heads = self.language_model_config.n_kv_heads
        head_size = self.language_model_config.head_dim
        total_memory = torch.cuda.get_device_properties(torch.device(device)).total_memory
        model_params = self.model_profiler.profile_model_params()
        kv_cache_size = int((model_params.vision_model_params + model_params.language_model_parmas) * get_dtype_size(dtype) * 0.8)
        n_blocks = kv_cache_size // n_layers // n_tokens // block_size // n_kv_heads // head_size
        kv_caches = [KVCache(
            key_cache   = torch.randn(size=(n_blocks, block_size, n_kv_heads, head_size), dtype=dtype, device=device), 
            value_cache = torch.randn(size=(n_blocks, block_size, n_kv_heads, head_size), dtype=dtype, device=device), 
        ) for _ in range(n_layers)] 

        builder = AttentionParametersBuilder(
            num_qo_heads=self.language_model_config.n_qo_heads, 
            num_kv_heads=self.language_model_config.n_kv_heads, 
            head_dim=self.language_model_config.head_dim, 
            block_size=block_size, 
            device=device, 
        )
        for kv_cache in kv_caches:
            builder.add_kv_cache(kv_cache)

        self.has_language_model_task = False

        all_seqeunce_decode = True
        for request_meta_data in self.requests_meta_data:
            if not (request_meta_data.stage == 'prefill' or request_meta_data.stage == 'decode'):
                continue
            self.has_language_model_task = True
            if request_meta_data.stage == 'prefill':
                all_seqeunce_decode = False
            q_seq_len = request_meta_data.n_prompt_tokens if request_meta_data.stage == 'prefill' else 1
            kv_seq_len = 0 if request_meta_data.stage == 'prefill' else request_meta_data.n_kv_cache_tokens
            new_cache_slots = list(range(request_meta_data.n_prompt_tokens)) if request_meta_data.stage == 'prefill' else [kv_seq_len + 1]
            block_table = list(range((kv_seq_len + 1 + block_size - 1) // block_size))
            for i in range(request_meta_data.n_requests):
                input_ids_list.extend(list(range(q_seq_len)))
                position_ids_list.extend(list(range(q_seq_len)))
                selected_token_ids_list.append(len(input_ids_list) - 1)
                builder.add_request(
                    q_seq_len=q_seq_len, 
                    kv_seq_len=kv_seq_len, 
                    new_cache_slots=new_cache_slots, 
                    block_table=block_table,
                )
        self.input_ids = torch.tensor(input_ids_list, dtype=torch.int, device=device)
        self.position_ids = torch.tensor(position_ids_list, dtype=torch.int, device=device)
        self.selected_token_ids = torch.tensor(selected_token_ids_list, dtype=torch.int, device=device)
        self.attention_params = builder.build_attention_parameters()
        self.model_params = LanguageModelParameters(
            attention_params = self.attention_params, 
            all_sequences_decode = all_seqeunce_decode, 
            selected_token_ids = self.selected_token_ids, 
        )

    def __call__(self):
        if self.has_vision_model_task:
            self.vision_model.forward(self.pixel_values, self.vision_model_params)

        if self.has_language_model_task:
            self.language_model.forward(
                input_ids = self.input_ids, 
                image_features = None, 
                position_ids = self.position_ids, 
                model_params = self.model_params, 
            )


if __name__ == '__main__':
    factory = getModelFactory(ModelFactoryConfig(path='/mnt/cfs/9n-das-admin/llm_models/llava-1.5-7b-hf'), ModelFactoryContext())
    language_model = factory.getLanguageModel()
    vision_model = factory.getVisionModel()
    model_profiler = factory.getModelProfiler()
    vision_model_config = factory.getVisionModelConfig()
    language_model_config = factory.getLanguageModelConfig()
    processor = factory.getProcessor()

    max_batch_size = 11
    for batch_size in range(1, max_batch_size + 1):
        op = Batch(
            vision_model = vision_model, 
            language_model = language_model, 
            vision_model_config = vision_model_config, 
            language_model_config = language_model_config, 
            model_profiler = model_profiler, 
            processor = processor, 
            requests_meta_data = [
                RequestMetaData(
                    n_requests = 2 ** batch_size,
                    stage = 'decode', 
                    image_sizes = [(336, 336)],
                    n_prompt_tokens = 1,
                    n_kv_cache_tokens = 1024,
                )
            ]
        )
        from batch_analysis import benchmark
        latency = benchmark(op)
        # throughput = batch_size * 576 / latency
        print(f'batch_size {2 ** batch_size} {latency} s')