import random
import flashinfer
import copy
from itertools import accumulate
import math
import torch
from torch import Tensor
import pytest
from flash_attn import flash_attn_func, flash_attn_qkvpacked_func, flash_attn_with_kvcache, flash_attn_kvpacked_func
from dxz._C.kernel.flash_attn import mha_varlen_fwd
from dxz.memory.kv_cache import KVCache
from dxz.memory.block_allocator import BlockAllocator
from dxz.layer.attention import TorchCausalGroupedQueryPageAttention
from dxz.model.parameters import InputParameters

@pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("seq_len", [16, 576, 577])
@pytest.mark.parametrize("num_heads", [(8, 8), (8, 4), (8, 1)])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("device", [torch.device('cuda:0')])
@torch.inference_mode()
def test_flash_attn_func(
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

@pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("num_heads", [(8, 8), (8, 4), (8, 1)])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("device", [torch.device('cuda:0')])
@torch.inference_mode()
def test_flash_infer_batch_decode(
    batch_size: int, 
    num_heads: tuple[int, int], 
    head_dim: int, 
    causal: bool,
    dtype: torch.dtype,
    device: torch.device,
):
    n_tokens = batch_size
    num_qo_heads, num_kv_heads = num_heads
    total_blocks = max_num_blocks = max_num_pages = 128
    block_size = page_size = 16

    q = torch.randn(size=(batch_size, num_qo_heads, head_dim), dtype=dtype, device=device)
    block_tables = kv_page_indices = torch.randperm(total_blocks, dtype=torch.int, device=device)
    q_cu_seq_lens = torch.arange(batch_size + 1, dtype=torch.int, device=device)
    k_seq_lens = [random.randint(1, 256) for _ in range(batch_size)]
    k_cu_seq_lens = torch.tensor(list(accumulate(k_seq_lens, initial=0)), dtype=torch.int, device=device)
    cu_block_lens = kv_page_indptr = torch.tensor(list(accumulate([(k_seq_len + block_size - 1) // block_size for k_seq_len in k_seq_lens], initial=0)), dtype=torch.int32, device=device)
    kv_last_page_len = torch.tensor([page_size if k_seq_len % page_size == 0 else k_seq_len % page_size for k_seq_len in k_seq_lens], dtype=torch.int, device=device)

    kvcache = KVCache(max_num_pages, page_size, num_kv_heads, head_dim, dtype, device)
    k_cache = torch.randn(size=(max_num_pages, page_size, num_kv_heads, head_dim), dtype=dtype, device=device)
    v_cache = torch.randn(size=(max_num_pages, page_size, num_kv_heads, head_dim), dtype=dtype, device=device)

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "NHD")
    decode_wrapper.plan(
        kv_page_indptr,
        kv_page_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        data_type=dtype
    )

    o = decode_wrapper.run(q, (k_cache, v_cache))

    o_ref = torch.zeros(size=(batch_size, num_qo_heads, head_dim), dtype=dtype, device=device)
    mha_varlen_fwd(
        o_ref, 
        q,
        k_cache,
        v_cache,
        q_cu_seq_lens, 
        k_cu_seq_lens, 
        block_tables, 
        cu_block_lens, 
        None, 
        1, 
        max(k_seq_lens), 
        1. / math.sqrt(head_dim), 
        0.,
        -1, 
        0, 
        0, 
    )

    print(f'o.view(-1)[:8] {o.view(-1)[:8]}')
    print(f'o_ref.view(-1)[:8] {o_ref.view(-1)[:8]}')
    assert torch.allclose(o, o_ref, atol=1e-3, rtol=1e-3)

@pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("num_heads", [(8, 8), (8, 4), (8, 1)])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("device", [torch.device('cuda:0')])
@torch.inference_mode()
def test_flash_infer_batch_prefill(
    batch_size: int, 
    num_heads: tuple[int, int], 
    head_dim: int, 
    causal: bool,
    dtype: torch.dtype,
    device: torch.device,
):
    import flashinfer
    num_qo_heads, num_kv_heads = num_heads
    total_blocks = max_num_blocks = max_num_pages = 128
    block_size = page_size = 16

    q_seq_lens = [random.randint(1, 20) for _ in range(batch_size)]
    q_cu_seq_lens = qo_indptr = torch.tensor(list(accumulate(q_seq_lens, initial=0)), dtype=torch.int, device=device)
    k_seq_lens = [random.randint(1, 256) for _ in range(batch_size)]
    k_cu_seq_lens = torch.tensor(list(accumulate(k_seq_lens, initial=0)), dtype=torch.int, device=device)
    paged_kv_last_page_len = torch.tensor([page_size if k_seq_len % page_size == 0 else k_seq_len % page_size for k_seq_len in k_seq_lens], dtype=torch.int32, device=device)

    block_tables = paged_kv_indices = torch.arange(max_num_pages, dtype=torch.int, device=device)
    cu_block_lens = paged_kv_indptr = torch.tensor(list(accumulate([((k_seq_len + block_size - 1) // block_size) for k_seq_len in k_seq_lens], initial=0)), dtype=torch.int32, device=device)

    q = torch.randn(size=(sum(q_seq_lens), num_qo_heads, head_dim), dtype=dtype, device=device)
    k_cache = torch.randn(size=(max_num_pages, page_size, num_kv_heads, head_dim), dtype=dtype, device=device)
    v_cache = torch.randn(size=(max_num_pages, page_size, num_kv_heads, head_dim), dtype=dtype, device=device)

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, "NHD")
    prefill_wrapper.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=causal,
    )

    o = prefill_wrapper.run(q, (k_cache, v_cache))
    o_ref = torch.zeros(size=(sum(q_seq_lens), num_qo_heads, head_dim), dtype=dtype, device=device)
    mha_varlen_fwd(
        o_ref, 
        q, 
        k_cache, 
        v_cache, 
        q_cu_seq_lens, 
        k_cu_seq_lens, 
        block_tables, 
        cu_block_lens, 
        None, 
        max(q_seq_lens), 
        max(k_seq_lens), 
        1. / math.sqrt(head_dim), 
        0., 
        -1, 
        0, 
        0, 
    )

    print(f'o.view(-1)[:8] {o.view(-1)[:8]}')
    print(f'o_ref.view(-1)[:8] {o_ref.view(-1)[:8]}')
    assert torch.allclose(o, o_ref, atol=1e-3, rtol=1e-3)

if __name__ == '__main__':
    pytest.main([__file__ + '::test_flash_attn_func', '-x'])
    pytest.main([__file__ + '::test_flash_infer_batch_decode', '-x'])
    pytest.main([__file__ + '::test_flash_infer_batch_prefill', '-x'])