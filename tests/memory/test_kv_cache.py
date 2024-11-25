import torch
import pytest
from dxz.memory.kv_cache import KVCache
import copy

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
    kv_cache = KVCache(num_blocks, block_size, num_kv_heads, head_size, dtype, device)
    kv_cache_ref = copy.deepcopy(kv_cache)
    key_cache, value_cache = kv_cache.get_kv_cache()
    key_cache_ref, value_cache_ref = kv_cache_ref.get_kv_cache()

    slot_ids = torch.randperm(num_blocks, dtype=torch.int, device=device)[:n_tokens] # random select n_tokens different number in [0, num_blocks)
    key   = torch.randn(size=(n_tokens, num_kv_heads, head_size), dtype=dtype, device=device)
    value = torch.randn(size=(n_tokens, num_kv_heads, head_size), dtype=dtype, device=device)
    kv_cache.set_kv_cache(slot_ids, key, value)
    
    for i in range(n_tokens):
        slot_id = slot_ids[i]
        block_id = slot_id // block_size
        block_offset = slot_id % block_size
        key_cache_ref[block_id, block_offset, :, :] = key[i, :, :]
        value_cache_ref[block_id, block_offset, :, :] = value[i, :, :]
    
    assert torch.allclose(key_cache, key_cache_ref)
    assert torch.allclose(value_cache, value_cache_ref)


@pytest.mark.parametrize("num_blocks", [100])
@pytest.mark.parametrize("block_size", [4, 8, 16])
@pytest.mark.parametrize("num_kv_heads", [8, 4, 2, 1])
@pytest.mark.parametrize("head_size", [64, 128, 256])
@pytest.mark.parametrize("n_tokens", [15, 64, 100])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float])
@pytest.mark.parametrize("device", [torch.device('cpu'), torch.device('cuda:0')])
@torch.inference_mode()
def test_move_kv_cache(
    num_blocks: int, 
    block_size: int, 
    num_kv_heads: int, 
    head_size: int, 
    n_tokens: int,
    dtype: torch.dtype, 
    device: torch.device
):
    kv_cache = KVCache(num_blocks, block_size, num_kv_heads, head_size, dtype, device)
    kv_cache_ref = copy.deepcopy(kv_cache)
    kv_cache_ref_output = copy.deepcopy(kv_cache)

    key_cache, value_cache = kv_cache.get_kv_cache()
    key_cache_ref, value_cache_ref = kv_cache_ref.get_kv_cache()
    key_cache_ref_output, value_cache_ref_output = kv_cache_ref_output.get_kv_cache()


    total_slots = num_blocks * block_size
    src_slot_ids = torch.randperm(total_slots, dtype=torch.int, device=device)[:n_tokens] # [0, total_slots)
    dst_slot_ids = torch.randperm(total_slots, dtype=torch.int, device=device)[:n_tokens] # [0, total_slots)

    assert torch.allclose(key_cache, key_cache_ref)
    assert torch.allclose(value_cache, value_cache_ref)
    assert torch.allclose(key_cache, key_cache_ref_output)
    assert torch.allclose(value_cache, value_cache_ref_output)

    kv_cache.move_kv_cache(src_slot_ids, dst_slot_ids)

    assert not torch.allclose(key_cache, key_cache_ref)
    assert not torch.allclose(value_cache, value_cache_ref)
    assert not torch.allclose(key_cache, key_cache_ref_output)
    assert not torch.allclose(value_cache, value_cache_ref_output)

    for src_slot_id, dst_slot_id in zip(src_slot_ids.tolist(), dst_slot_ids.tolist()):
        key_cache_ref_output.view(total_slots, -1)[dst_slot_id] = key_cache_ref.view(total_slots, -1)[src_slot_id]
        value_cache_ref_output.view(total_slots, -1)[dst_slot_id] = value_cache_ref.view(total_slots, -1)[src_slot_id]

    assert not torch.allclose(key_cache, key_cache_ref)
    assert not torch.allclose(value_cache, value_cache_ref)
    assert torch.allclose(key_cache, key_cache_ref_output)
    assert torch.allclose(value_cache, value_cache_ref_output)


if __name__ == '__main__':
    pytest.main([__file__, '--maxfail=1'])