import copy
from itertools import accumulate
import math
import torch
from torch import Tensor
import pytest
from flash_attn import flash_attn_func, flash_attn_qkvpacked_func, flash_attn_with_kvcache, flash_attn_kvpacked_func
from dxz.memory.kv_cache import KVCache
from dxz.memory.block_allocator import BlockAllocator
from dxz.layer.attention import TorchCausalGroupedQueryPageAttention
from dxz.model.parameters import InputParameters

"""
if q fixed length:
    if k fixed length(no kvcache):
        flash_attn_qkvpacked_func
        flash_attn_kvpacked_func
        flash_attn_func
    elif kvcache:
        flash_attn_with_kvcache
elif q variable length:
    if k variable length(nokvcache):
        flash_attn_varlen_qkvpacked_func
        flash_attn_varlen_kvpacked_func
        flash_attn_varlen_func
    elif kvcache:
        flash_attn_with_kvcache
"""

@pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("seq_len", [16, 576, 577])
@pytest.mark.parametrize("num_heads", [(8, 8), (8, 4), (8, 1)])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("device", [torch.device('cuda:0')])
@torch.inference_mode()
def test_fixedlength_nokvcache_attention(
    batch_size: int, 
    seq_len: int, 
    num_heads: tuple[int, int], 
    head_dim: int, 
    causal: bool,
    dtype: torch.dtype,
    device: torch.device,
):
    group_size = num_heads[0] // num_heads[1]

    q = torch.randn(size=(batch_size, seq_len, num_heads[0], head_dim), dtype=dtype, device=device)
    k = torch.randn(size=(batch_size, seq_len, num_heads[1], head_dim), dtype=dtype, device=device)
    v = torch.randn(size=(batch_size, seq_len, num_heads[1], head_dim), dtype=dtype, device=device)
    def ref_attention(q, k, v):
        k = torch.repeat_interleave(k, repeats=group_size, dim=2)
        v = torch.repeat_interleave(v, repeats=group_size, dim=2)
        q = q.transpose(1, 2).contiguous().view(-1, seq_len, head_dim).to(torch.float)
        k = k.transpose(1, 2).contiguous().view(-1, seq_len, head_dim).to(torch.float)
        v = v.transpose(1, 2).contiguous().view(-1, seq_len, head_dim).to(torch.float)
        scores = torch.bmm(q, k.transpose(-1, -2)) * 1. / math.sqrt(head_dim)
        if causal:
            mask = torch.tril(torch.ones(size=(seq_len, seq_len), dtype=torch.bool, device=device))
            scores = torch.masked_fill(input=scores, mask=~mask, value=float('-inf'))
        scores_presoftmax = scores
        scores = torch.softmax(scores, dim=-1)
        o = torch.bmm(scores, v)
        o = o.reshape(batch_size, num_heads[0], seq_len, head_dim)
        o = o.transpose(1, 2).contiguous().to(dtype)
        return o, scores_presoftmax.to(dtype), scores.to(dtype)
    out_ref, scores_presoftmax, scores = ref_attention(q, k, v)

    flash_attn_func_out, flash_attn_func_softmax_lse, flash_attn_func_S_dmask = flash_attn_func(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=None,
        causal=causal,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=True,
    )
    print(f'scores_presoftmax.shape {scores_presoftmax.shape}')
    print(f'scores.shape {scores.shape}')
    print(f'flash_attn_func_softmax_lse.shape {flash_attn_func_softmax_lse.shape}')
    print(f'flash_attn_func_S_dmask.shape {flash_attn_func_S_dmask.shape}')
    # assert torch.allclose(flash_attn_func_S_dmask, scores, rtol=1e-3, atol=1e-3)
    assert torch.allclose(flash_attn_func_out, out_ref, rtol=1e-3, atol=1e-3)

    if group_size == 1:
        qkv = torch.cat([q[:, :, None, :, :], k[:, :, None, :, :], v[:, :, None, :, :]], dim=2)
        flash_attn_qkvpacked_func_out = flash_attn_qkvpacked_func(
            qkv,
            dropout_p=0.0,
            softmax_scale=None,
            causal=causal,
            window_size=(-1, -1),  # -1 means infinite context window
            softcap=0.0,  # <=0.0 means deactivate
            alibi_slopes=None,
            deterministic=False,
            return_attn_probs=False,
        )
        assert torch.allclose(flash_attn_qkvpacked_func_out, out_ref, rtol=1e-3, atol=1e-3)

    kv = torch.cat([k[:, :, None, :, :], v[:, :, None, :, :]], dim=2)
    flash_attn_kvpacked_func_out = flash_attn_kvpacked_func(
        q,
        kv,
        dropout_p=0.0,
        softmax_scale=None,
        causal=causal,
        window_size=(-1, -1),  # -1 means infinite context window
        softcap=0.0,  # 0.0 means deactivated
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
    )
    assert torch.allclose(flash_attn_kvpacked_func_out, out_ref, rtol=1e-3, atol=1e-3)

# @pytest.mark.parametrize("seq_lens", [4, 16, 576, 577])
# @pytest.mark.parametrize("num_heads", [(8, 8), (8, 4), (8, 1)])
# @pytest.mark.parametrize("head_dim", [64, 128, 256])
# @pytest.mark.parametrize("causal", [False, True])
# @pytest.mark.parametrize("dtype", [torch.float16])
# @pytest.mark.parametrize("device", [torch.device('cuda:0')])
# @torch.inference_mode()
# def test_varlength_nokvcache_attention(
#     seq_lens: list[int], 
#     num_heads: tuple[int, int], 
#     head_dim: int, 
#     causal: bool,
#     dtype: torch.dtype,
#     device: torch.device,
# ):
#     group_size = num_heads[0] // num_heads[1]

#     sum_seq_lens = sum(seq_lens)
#     q = torch.randn(size=(sum_seq_lens, num_heads[0], head_dim), dtype=dtype, device=device)
#     k = torch.randn(size=(sum_seq_lens, num_heads[1], head_dim), dtype=dtype, device=device)
#     v = torch.randn(size=(sum_seq_lens, num_heads[1], head_dim), dtype=dtype, device=device)

# @pytest.mark.parametrize("seq_lens", [[(1, 100), (15, 15), (111, 234), (1000, 10000)]])
# @pytest.mark.parametrize("num_heads", [(8, 8), (8, 4), (8, 2), (8, 1)])
# @pytest.mark.parametrize("head_size", [64, 128, 256])
# @pytest.mark.parametrize("n_blocks", [100])
# @pytest.mark.parametrize("block_size", [4, 8, 16])
# @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
# @pytest.mark.parametrize("logits_soft_cap", [0.0, 50.0])
# @pytest.mark.parametrize("sliding_window", [-1, 50])
# @pytest.mark.parametrize("alibi", [False, True])

# @pytest.mark.parametrize("seq_lens", [[(1, 100), (1, 15), (1, 234), (1, 10000)]])
# @pytest.mark.parametrize("num_heads", [(8, 8)])
# @pytest.mark.parametrize("head_size", [64])
# @pytest.mark.parametrize("n_blocks", [100])
# @pytest.mark.parametrize("block_size", [16])
# @pytest.mark.parametrize("dtype", [torch.float16])
# @pytest.mark.parametrize("device", [torch.device('cuda:0')])
# @torch.inference_mode()
# def test_fixedlength_kvcache_attention(
#     seq_lens  : list[tuple[int, int]], # each tuple is a sequence
#     num_heads : list[tuple[int]], # q heads and kv heads
#     head_size : int, 
#     n_blocks  : int,
#     block_size: int,
#     dtype     : torch.dtype, 
#     device    : torch.device,
# ):
#     q_seq_lens = [x for x, _ in seq_lens]

#     sum_q_seq_len = sum(q_seq_lens)
#     kv_seq_lens = [x for _, x in seq_lens]
#     block_lens = [(kv_seq_len + block_size - 1) // block_size for kv_seq_len in kv_seq_lens]
#     allocator = BlockAllocator(n_blocks)
#     block_tables: list[list[int]] = []
#     for block_len in block_lens:
#         block_tables.append(allocator.allocate(block_len))

#     q = torch.randn(size=(sum_q_seq_len, num_heads[0], head_size), dtype=dtype, device=device)
#     k = torch.randn(size=(sum_q_seq_len, num_heads[1], head_size), dtype=dtype, device=device)
#     v = torch.randn(size=(sum_q_seq_len, num_heads[1], head_size), dtype=dtype, device=device)
#     new_cache_slots = torch.randperm(n_blocks * block_size, dtype=torch.int, device=device)[: sum_q_seq_len], 
#     kv_cache = KVCache(num_blocks=n_blocks, block_size=block_size, num_kv_heads=num_heads[1], head_size=head_size, dtype=dtype, device=device)
#     kv_cache_ref = copy.deepcopy(kv_cache)


#     input_parmas = InputParameters(
#         num_sequences = len(seq_lens), 
#         q_cu_seq_lens = torch.tensor(list(accumulate(q_seq_lens, initial=0)), dtype=torch.int, device=device), 
#         kv_cu_seq_lens = torch.tensor(list(accumulate(kv_seq_lens, initial=0)), dtype=torch.int, device=device), 
#         new_cache_slots = new_cache_slots, 
#         block_tables = torch.tensor(list(range(n_blocks)), dtype=torch.int, device=device), 
#         cu_blocks_lens = torch.tensor(list(accumulate(block_lens, initial=0)), dtype=torch.int, device=device), 
#         q_max_seq_len = 1024, 
#         kv_max_seq_len = 1024, 
#     )
#     ref_attention = TorchCausalGroupedQueryPageAttention(
#         n_qo_heads = num_heads[0], 
#         n_kv_heads = num_heads[1], 
#         head_dim   = head_size, 
#     )

#     assert torch.allclose(kv_cache.key_cache, kv_cache_ref.key_cache, rtol=1e-3, atol=1e-3)
#     assert torch.allclose(kv_cache.value_cache, kv_cache_ref.value_cache, rtol=1e-3, atol=1e-3)
#     out_ref = ref_attention(q, k, v, kv_cache_ref, input_parmas)
#     assert not torch.allclose(kv_cache.key_cache, kv_cache_ref.key_cache, rtol=1e-3, atol=1e-3)
#     assert not torch.allclose(kv_cache.value_cache, kv_cache_ref.value_cache, rtol=1e-3, atol=1e-3)

    # flash_attn_with_kvcache_out = flash_attn_with_kvcache(
    #     q = q,
    #     k_cache = kv_cache.key_cache,
    #     v_cache = kv_cache.value_cache,
    #     k = k,
    #     v = v,
    #     rotary_cos = None,
    #     rotary_sin = None,
    #     cache_seqlens = torch.tensor(kv_seq_lens, dtype=torch.int, device=device), 
    #     cache_batch_idx = new_cache_slots, 
    #     cache_leftpad = None,
    #     block_table: Optional[torch.Tensor] = None,
    #     softmax_scale=None,
    #     causal=False,
    #     window_size=(-1, -1),  # -1 means infinite context window
    #     softcap=0.0, # 0.0 means deactivated
    #     rotary_interleaved=True,
    #     alibi_slopes=None,
    #     num_splits=0,
    #     return_softmax_lse=False,
    # )

    # assert torch.allclose(kv_cache.key_cache, kv_cache_ref.key_cache, rtol=1e-3, atol=1e-3)
    # assert torch.allclose(kv_cache.value_cache, kv_cache_ref.value_cache, rtol=1e-3, atol=1e-3)

if __name__ == '__main__':
    pytest.main([__file__, '-x'])
    # pytest.main([__file__ + '::test_variable_length_attention', '-x'])
    # pytest.main([__file__, '-s']) # print