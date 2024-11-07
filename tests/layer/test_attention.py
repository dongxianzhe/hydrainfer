import torch
from dxz.memory.kv_cache import KVCache
from dxz.model.parameters import InputParameters
from typing import List, Tuple
from itertools import accumulate
import pytest
import copy

@pytest.mark.parametrize("batch_size", [1, 4, 15, 16])
@pytest.mark.parametrize("seq_len", [1, 14, 111, 576])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("n_heads", [8, 4, 2, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", [torch.device('cuda:0')])
@torch.inference_mode()
def test_attention(
    batch_size: int, 
    seq_len: int, 
    head_dim: int, 
    n_heads: int, 
    dtype: torch.dtype, 
    device: torch.device
):
    from dxz.layer.attention import TorchMultiHeadAttention
    from dxz.layer.attention import FlashMultiHeadAttention
    hidden_size = n_heads * head_dim

    attention_ref = TorchMultiHeadAttention(n_heads=n_heads, head_dim=head_dim)
    attention = FlashMultiHeadAttention(n_heads=n_heads, head_dim=head_dim)

    q = torch.randn(size=(batch_size, seq_len, hidden_size), dtype=dtype, device=device)
    k = torch.randn(size=(batch_size, seq_len, hidden_size), dtype=dtype, device=device)
    v = torch.randn(size=(batch_size, seq_len, hidden_size), dtype=dtype, device=device)

    o = attention(q, k, v)
    o_ref = attention_ref(q, k, v)
    print(f'o.view(-1)[:10]    : {o.view(-1)[:10]}')
    print(f'o_ref.view(-1)[:10]: {o_ref.view(-1)[:10]}')
    assert o.shape == o_ref.shape
    assert torch.allclose(o, o_ref, atol=1e-2, rtol=1e-2)

@pytest.mark.parametrize("seq_lens", [[(1, 100), (15, 15), (111, 234), (1, 1024)]])
@pytest.mark.parametrize("num_heads", [(8, 8), (8, 4), (8, 2), (8, 1)])
@pytest.mark.parametrize("head_size", [64, 128, 256])
@pytest.mark.parametrize("n_blocks", [10000])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", [torch.device('cuda:0')])
@torch.inference_mode
def test_causal_attention(
    seq_lens: List[Tuple[int, int]],
    num_heads: Tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    n_blocks: int,
    block_size: int,
    device: torch.device
):
    from dxz.layer.attention import TorchCausalGroupedQueryPageAttention, FlashCausalGroupedQueryPageAttention
    from dxz.memory.block_allocator import BlockAllocator
    # compute some mesc things
    seed = 42  
    torch.manual_seed(seed)  
    n_qo_heads, n_kv_heads = num_heads
    # create model and input and model_ref and input_ref
    attention = TorchCausalGroupedQueryPageAttention(n_qo_heads, n_kv_heads, head_size).to(device)
    flashattention = FlashCausalGroupedQueryPageAttention(n_qo_heads, n_kv_heads, head_size).to(device)

    q_seq_lens, kv_seq_lens = [seq_len[0] for seq_len in seq_lens], [seq_len[1] for seq_len in seq_lens]
    n_tokens, n_kv_cache_tokens = sum(q_seq_lens), sum(kv_seq_lens)
    query = torch.randn(n_tokens, head_size * n_qo_heads, dtype=dtype, device=device)
    key   = torch.randn(n_tokens, head_size * n_kv_heads, dtype=dtype, device=device)
    value = torch.randn(n_tokens, head_size * n_kv_heads, dtype=dtype, device=device)

    new_cache_slots = []
    allocator = BlockAllocator(n_blocks)
    block_tables = []
    cu_blocks_lens = [0]
    for q_seq_len, kv_seq_len in zip(q_seq_lens, kv_seq_lens):
        n_blocks_allocate = (kv_seq_len + block_size - 1) // block_size
        block_table = allocator.allocate(n_blocks_allocate)
        for i in range(kv_seq_len - q_seq_len, kv_seq_len):
            new_cache_slots.append(block_table[i // block_size] * block_size + i % block_size)
        block_tables += block_table
        cu_blocks_lens.append(cu_blocks_lens[-1] + n_blocks_allocate)

    kv_cache     = KVCache(n_blocks, block_size, n_kv_heads, head_size, dtype=dtype, device=device)
    kv_cache_ref = copy.deepcopy(kv_cache)

    input_params = InputParameters(
        num_sequences = len(seq_lens), 
        q_cu_seq_lens = torch.tensor([0] + list(accumulate(q_seq_lens)), dtype=torch.int ,device=device),
        kv_cu_seq_lens = torch.tensor([0] + list(accumulate(kv_seq_lens)), dtype=torch.int ,device=device),
        new_cache_slots = torch.tensor(new_cache_slots, dtype=torch.int, device=device),
        block_tables = torch.tensor(block_tables, dtype=torch.int, device=device), 
        cu_blocks_lens = torch.tensor(cu_blocks_lens, dtype=torch.int, device=device)
    )
    # forward
    output = flashattention(query, key, value, kv_cache, input_params)
    output_ref = attention(query, key, value, kv_cache_ref, input_params)
    # compare result
    assert torch.allclose(kv_cache.key_cache  , kv_cache_ref.key_cache  , atol=1e-2, rtol=1e-2), 'key cache is wrong'
    assert torch.allclose(kv_cache.value_cache, kv_cache_ref.value_cache, atol=1e-2, rtol=1e-2), 'value cache is wrong'
    print(output_ref.view(-1)[:10])
    print(output.view(-1)[:10])
    assert torch.allclose(output_ref, output, atol=1e-2, rtol=1e-2), 'output wrong'

if __name__ == '__main__':
    pytest.main([__file__])