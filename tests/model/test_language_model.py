import torch
from torch import Tensor
import pytest
from hydrainfer.memory.kv_cache import KVCache
from hydrainfer.model import getModelFactory, ModelFactoryConfig, ModelFactoryContext, ImageProcessor, VisionModelParameters, VisionModelOutput
from hydrainfer.utils.torch_utils import str2device, str2dtype
from hydrainfer.model.parameters import LanguageModelOutput, LanguageModelParameters
from hydrainfer.layer.causal_attention import AttentionParametersBuilder
from PIL import Image
import numpy as np


@pytest.mark.parametrize('path', [
    "/models/llava-1.5-7b-hf", 
    "/models/llava-v1.6-vicuna-7b-hf", 
    "/models/Qwen2-VL-7B/models--Qwen--Qwen2-VL-7B/snapshots/e61834264a23db10c06dc4f566dac5634c7ca024", 
    "/models/deepseek-vl2-tiny", 
    "/models/OpenGVLab/InternVL2-26B", 
])
def test_language_model(path: str):
    str_device = 'cuda:0'
    str_dtype = 'fp16'
    device = str2device(str_device)
    dtype = str2dtype(str_dtype)
    print(f'============================================================')
    print(f'model {path}')
    factory = getModelFactory(ModelFactoryConfig(path=path, device=str_device, dtype=str_dtype), ModelFactoryContext())
    language_model = factory.getLanguageModel()
    language_config = factory.getLanguageModelConfig()
    tokenizer = factory.getTokenizer()

    prompt = "hello world!"
    token_ids = tokenizer.encode(prompt)
    n_tokens = len(token_ids)
    input_ids = torch.tensor(token_ids, dtype=torch.int, device=device)

    n_blocks = 100
    block_size = 16
    kv_caches = [KVCache(
        torch.randn(size=(n_blocks, block_size, language_config.n_kv_heads, language_config.head_dim), dtype=dtype, device=device), 
        torch.randn(size=(n_blocks, block_size, language_config.n_kv_heads, language_config.head_dim), dtype=dtype, device=device), 
    ) for layer_id in range(language_config.n_layers)]

    builder = AttentionParametersBuilder(num_qo_heads=language_config.n_qo_heads, num_kv_heads=language_config.n_kv_heads, head_dim=language_config.head_dim, block_size=block_size, device=device)
    for kv_cache in kv_caches:
        builder.add_kv_cache(kv_cache)
    builder.add_request(n_tokens, kv_seq_len=0, new_cache_slots=list(range(n_tokens)), block_table=list(range((n_tokens + block_size - 1) // block_size)))

    output = language_model.forward(
        input_ids=input_ids, 
        image_features=None, 
        position_ids=torch.arange(n_tokens, dtype=torch.int, device=device), 
        model_params=LanguageModelParameters(
            attention_params = builder.build_attention_parameters(), 
            all_sequences_decode = False, 
            selected_token_ids = [n_tokens - 1]
        )
    )
    print(f'output.sample_token_ids.shape {output.sample_token_ids.shape}')
    assert output.sample_token_ids.shape == (1, )


if __name__ == '__main__':
    pytest.main([__file__, '-s'])