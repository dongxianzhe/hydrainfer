import copy
import torch
import pytest
from typing import Tuple, List
from itertools import accumulate
from hydrainfer.memory.kv_cache import KVCache
from hydrainfer.layer.causal_attention import AttentionParameters
from hydrainfer.layer.causal_attention import TorchCausalGroupedQueryPageAttentionHandler, FlashInferCausalGroupedQueryPageAttentionHandler, CausalGroupedQueryPageAttentionConfig, FlashAttentionCausalGroupedQueryPageAttentionHandler, AttentionParametersBuilder


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
    from hydrainfer.memory.block_allocator import BlockAllocator
    # compute some mesc things
    seed = 42  
    torch.manual_seed(seed)  
    n_qo_heads, n_kv_heads = num_heads

    config = CausalGroupedQueryPageAttentionConfig(
        n_qo_heads = n_qo_heads, 
        n_kv_heads = n_kv_heads, 
        head_dim = head_size
    )
    # create model and input and model_ref and input_ref
    attention = TorchCausalGroupedQueryPageAttentionHandler(config)
    flashattention = FlashAttentionCausalGroupedQueryPageAttentionHandler(config)

    q_seq_lens, kv_seq_lens = [seq_len[0] for seq_len in seq_lens], [seq_len[1] for seq_len in seq_lens]
    n_tokens, n_kv_cache_tokens = sum(q_seq_lens), sum(kv_seq_lens)
    query = torch.randn(n_tokens, n_qo_heads, head_size, dtype=dtype, device=device)
    key   = torch.randn(n_tokens, n_kv_heads, head_size, dtype=dtype, device=device)
    value = torch.randn(n_tokens, n_kv_heads, head_size, dtype=dtype, device=device)

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

    key_cache = torch.randn(size=(n_blocks, block_size, n_kv_heads, head_size), dtype=dtype, device=device)
    value_cache = torch.randn(size=(n_blocks, block_size, n_kv_heads, head_size), dtype=dtype, device=device)
    kv_cache     = KVCache(key_cache, value_cache)
    kv_cache_ref = copy.deepcopy(kv_cache)

    params = AttentionParameters(
        num_sequences = len(seq_lens), 
        q_cu_seq_lens = torch.tensor([0] + list(accumulate(q_seq_lens)), dtype=torch.int ,device=device),
        kv_cu_seq_lens = torch.tensor([0] + list(accumulate(kv_seq_lens)), dtype=torch.int ,device=device),
        new_cache_slots = torch.tensor(new_cache_slots, dtype=torch.int, device=device),
        block_tables = torch.tensor(block_tables, dtype=torch.int, device=device), 
        cu_blocks_lens = torch.tensor(cu_blocks_lens, dtype=torch.int, device=device), 
        kv_cache = kv_cache, 
    )
    output = flashattention(query, params).o
    output_ref = attention(query, params).o

    print(output.shape)
    print(output_ref.shape)
    print(output_ref.view(-1)[-10:])
    print(output.view(-1)[-10:])
    assert torch.allclose(kv_cache.key_cache  , kv_cache_ref.key_cache  , atol=1e-2, rtol=1e-2), 'key cache is wrong'
    assert torch.allclose(kv_cache.value_cache, kv_cache_ref.value_cache, atol=1e-2, rtol=1e-2), 'value cache is wrong'
    assert torch.allclose(output_ref, output, atol=1e-2, rtol=1e-2), 'output wrong'


if __name__ == '__main__':
    pytest.main([__file__, '-s'])