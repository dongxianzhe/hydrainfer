import torch
import pytest
from dxz.memory.kv_cache import KVCache

@pytest.mark.parametrize("num_blocks", [100])
@pytest.mark.parametrize("block_size", [4, 8, 16])
@pytest.mark.parametrize("num_kv_heads", [8, 4, 2, 1])
@pytest.mark.parametrize("head_size", [64, 128, 256])
@pytest.mark.parametrize("n_tokens", [1, 15, 64, 100])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float])
@pytest.mark.parametrize("device", [torch.device('cpu'), torch.device('cuda:0')])
@torch.inference_mode()
def test_set_kv_cache(
    num_blocks: int, 
    block_size: int, 
    num_kv_heads: int, 
    head_size: int, 
    n_tokens: int,
    dtype: torch.dtype, 
    device: torch.device
):
    kv_cache = KVCache(
        torch.zeros(size=(num_blocks, block_size, num_kv_heads, head_size), dtype=dtype, device=device), 
        torch.zeros(size=(num_blocks, block_size, num_kv_heads, head_size), dtype=dtype, device=device)
    )
    slot_ids = torch.randperm(num_blocks, dtype=torch.int, device=device)[:n_tokens] # random select n_tokens different number in [0, num_blocks)
    key   = torch.randn(size=(n_tokens, num_kv_heads, head_size), dtype=dtype, device=device)
    value = torch.randn(size=(n_tokens, num_kv_heads, head_size), dtype=dtype, device=device)
    kv_cache.set_kv_cache(slot_ids, key, value)

    key_cache, value_cache = kv_cache.get_kv_cache()

    key_cache_ref   = torch.zeros(size=(num_blocks, block_size, num_kv_heads, head_size), dtype=dtype, device=device) 
    value_cache_ref = torch.zeros(size=(num_blocks, block_size, num_kv_heads, head_size), dtype=dtype, device=device)
    for i in range(n_tokens):
        slot_id = slot_ids[i]
        block_id = slot_id // block_size
        block_offset = slot_id % block_size
        key_cache_ref[block_id, block_offset, :, :] = key[i, :, :]
        value_cache_ref[block_id, block_offset, :, :] = value[i, :, :]
    
    assert torch.allclose(key_cache, key_cache_ref)
    assert torch.allclose(value_cache, value_cache_ref)


if __name__ == '__main__':
    pytest.main([__file__, '--maxfail=1'])