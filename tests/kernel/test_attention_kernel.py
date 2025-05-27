import random
import flashinfer
import copy
from itertools import accumulate
import math
import torch
from torch import Tensor
import pytest
from flash_attn import flash_attn_func, flash_attn_qkvpacked_func, flash_attn_with_kvcache, flash_attn_kvpacked_func
from hydrainfer._C.kernel.flash_attn import mha_varlen_fwd
from hydrainfer.memory.kv_cache import KVCache
from hydrainfer.memory.block_allocator import BlockAllocator
from hydrainfer.layer.attention import TorchCausalGroupedQueryPageAttention
from hydrainfer.model.parameters import InputParameters

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

@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize("num_heads", [(8, 8), (8, 4), (8, 1)])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("device", [torch.device('cuda:0')])
@pytest.mark.parametrize("mask_type", ["causal", "random", "streamingllm"])
@torch.inference_mode()
def test_flash_infer_batch_prefill_mask(
    batch_size: int, 
    num_heads : tuple[int, int], 
    head_dim  : int,
    dtype     : torch.dtype, 
    device    : torch.device,
    mask_type : str, 
):
    import flashinfer
    num_qo_heads, num_kv_heads = num_heads
    total_blocks = max_num_blocks = max_num_pages = 128
    block_size = page_size = 16
    num_sequences = batch_size

    q_seq_lens = [random.randint(1, 20) for _ in range(batch_size)]
    q_cu_seq_lens = qo_indptr = torch.tensor(list(accumulate(q_seq_lens, initial=0)), dtype=torch.int, device=device)
    k_seq_lens = [random.randint(21, 256) for _ in range(batch_size)]
    k_cu_seq_lens = torch.tensor(list(accumulate(k_seq_lens, initial=0)), dtype=torch.int, device=device)
    paged_kv_last_page_len = torch.tensor([page_size if k_seq_len % page_size == 0 else k_seq_len % page_size for k_seq_len in k_seq_lens], dtype=torch.int32, device=device)
    
    block_tables = paged_kv_indices = torch.arange(max_num_pages, dtype=torch.int, device=device)
    cu_block_lens = paged_kv_indptr = torch.tensor(list(accumulate([((k_seq_len + block_size - 1) // block_size) for k_seq_len in k_seq_lens], initial=0)), dtype=torch.int32, device=device)
    
    q = torch.randn(size=(sum(q_seq_lens), num_qo_heads, head_dim), dtype=dtype, device=device)
    k_cache = torch.randn(size=(max_num_pages, page_size, num_kv_heads, head_dim), dtype=dtype, device=device)
    v_cache = torch.randn(size=(max_num_pages, page_size, num_kv_heads, head_dim), dtype=dtype, device=device)

    masks = []
    if mask_type == 'causal':
        # causal mask
        for q_seq_len, k_seq_len in zip(q_seq_lens, k_seq_lens):
            mask = torch.ones(q_seq_len, k_seq_len, dtype=torch.bool, device=q.device)
            mask = mask.tril(diagonal=k_seq_len - q_seq_len)
            masks.append(mask.reshape(-1))
    elif mask_type == 'random':
        # random mask
        for q_seq_len, k_seq_len in zip(q_seq_lens, k_seq_lens):
            mask = torch.randint(0, 2, (q_seq_len, k_seq_len), dtype=torch.bool, device=device)
            masks.append(mask.reshape(-1))
    elif mask_type == 'streamingllm':
        attention_sink = 1
        window_size = 3
        for i, (q_seq_len, k_seq_len) in enumerate(zip(q_seq_lens, k_seq_lens)):
            mask = torch.ones(size=(q_seq_len, k_seq_len), dtype=torch.bool, device=device)
            mask = torch.triu(mask, diagonal=k_seq_len - q_seq_len - window_size)
            mask = torch.tril(mask, diagonal=k_seq_len - q_seq_len)
            mask[:, :attention_sink] = 1

            mask = torch.randint(0, 2, (q_seq_len, k_seq_len), dtype=torch.bool, device=device)
            masks.append(mask.reshape(-1))

    
    custom_mask = torch.cat(masks, dim=0)

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, "NHD", use_cuda_graph=False)
    prefill_wrapper.plan(
        qo_indptr              = qo_indptr,
        paged_kv_indptr        = paged_kv_indptr,
        paged_kv_indices       = paged_kv_indices,
        paged_kv_last_page_len = paged_kv_last_page_len,
        num_qo_heads           = num_qo_heads,
        num_kv_heads           = num_kv_heads,
        head_dim               = head_dim,
        page_size              = page_size,
        # causal                 = True,
        custom_mask            = custom_mask, 
    )
    o = prefill_wrapper.run(q, (k_cache, v_cache))
    
    def ref_batch_prefill(
        n_qo_heads: int, 
        n_kv_heads: int, 
        head_dim: int, 
        query, 
        k_cache: Tensor, 
        v_cache: Tensor, 
        num_sequences: int, 
        q_cu_seq_lens: Tensor, 
        cu_block_lens: Tensor, 
        kv_cu_seq_lens: Tensor, 
        custom_mask: Tensor, 
        ) -> Tensor:
        outputs = []
        cu_mask_len: int = 0
        for i in range(num_sequences):
            block_table = block_tables[cu_block_lens[i]: cu_block_lens[i + 1]]
            key = k_cache[block_table, :, :, :].reshape(-1, n_kv_heads, head_dim)
            value = v_cache[block_table, :, :, :].reshape(-1, n_kv_heads, head_dim)
            k = key[: kv_cu_seq_lens[i + 1] - kv_cu_seq_lens[i], :, :].to(torch.float)
            v = value[: kv_cu_seq_lens[i + 1] - kv_cu_seq_lens[i],:, :].to(torch.float)
            q = query[q_cu_seq_lens[i]: q_cu_seq_lens[i + 1], :, :].to(torch.float)
            # q (qo_seq_len, n_qo_heads, head_dim)
            # k (kv_seq_len, n_kv_heads, head_dim)
            # v (kv_seq_len, n_kv_heads, head_dim)
            group_size = n_qo_heads // n_kv_heads
            k = k.repeat_interleave(repeats=group_size, dim=1)
            v = v.repeat_interleave(repeats=group_size, dim=1)
            # k (kv_seq_len, n_qo_heads, head_dim)
            # v (kv_seq_len, n_qo_heads, head_dim)

            # compute score
            scores = torch.einsum('qhd,khd->hqk', q, k)

            # sm_scale
            sm_scale = 1. / math.sqrt(head_dim)
            scores *= sm_scale

            # mask
            _, q_seq_len, k_seq_len = scores.shape
            mask = custom_mask[cu_mask_len: cu_mask_len + q_seq_len * k_seq_len].reshape(q_seq_len, k_seq_len)
            cu_mask_len += q_seq_len * k_seq_len
            scores.masked_fill_(mask=~mask, value=float('-inf'))

            # softmax
            scores = torch.softmax(scores, dim=-1)
            o = torch.einsum("hqk,khd->qhd", scores, v)
            outputs.append(o)

        output = torch.cat(outputs, dim=0)
        return output.to(query.dtype)

    o_ref = ref_batch_prefill(
        n_qo_heads = num_qo_heads, 
        n_kv_heads = num_kv_heads, 
        head_dim = head_dim, 
        query = q, 
        k_cache = k_cache,
        v_cache = v_cache,
        num_sequences = batch_size, 
        q_cu_seq_lens = q_cu_seq_lens, 
        cu_block_lens = cu_block_lens, 
        kv_cu_seq_lens = k_cu_seq_lens, 
        custom_mask = custom_mask, 
    )

    print(f'o.view(-1)[:8] {o.view(-1)[:]}')
    print(f'o_ref.view(-1)[:8] {o_ref.view(-1)[:]}')
    assert torch.allclose(o, o_ref, atol=1e-1, rtol=1e-1)


@torch.inference_mode()
def test_flash_infer_batch_prefill_mask_perf():
    import flashinfer
    batch_size = 8
    num_heads  = (32, 32)
    head_dim   = 128
    dtype      = torch.float16
    device     = torch.device('cuda:0')
    num_qo_heads, num_kv_heads = num_heads
    total_blocks = max_num_blocks = max_num_pages = 1024
    block_size = page_size = 16
    num_sequences = batch_size

    q_seq_lens = [random.randint(256, 512) for _ in range(batch_size)]
    q_cu_seq_lens = qo_indptr = torch.tensor(list(accumulate(q_seq_lens, initial=0)), dtype=torch.int, device=device)
    k_seq_lens = [random.randint(512, 1024) for _ in range(batch_size)]
    k_cu_seq_lens = torch.tensor(list(accumulate(k_seq_lens, initial=0)), dtype=torch.int, device=device)
    paged_kv_last_page_len = torch.tensor([page_size if k_seq_len % page_size == 0 else k_seq_len % page_size for k_seq_len in k_seq_lens], dtype=torch.int32, device=device)
    
    block_tables = paged_kv_indices = torch.arange(max_num_pages, dtype=torch.int, device=device)
    cu_block_lens = paged_kv_indptr = torch.tensor(list(accumulate([((k_seq_len + block_size - 1) // block_size) for k_seq_len in k_seq_lens], initial=0)), dtype=torch.int32, device=device)
    
    q = torch.randn(size=(sum(q_seq_lens), num_qo_heads, head_dim), dtype=dtype, device=device)
    k_cache = torch.randn(size=(max_num_pages, page_size, num_kv_heads, head_dim), dtype=dtype, device=device)
    v_cache = torch.randn(size=(max_num_pages, page_size, num_kv_heads, head_dim), dtype=dtype, device=device)

    full_masks = []
    for q_seq_len, k_seq_len in zip(q_seq_lens, k_seq_lens):
        mask = torch.ones(q_seq_len, k_seq_len, dtype=torch.bool, device=q.device)
        full_masks.append(mask.reshape(-1))
    full_masks = torch.cat(full_masks, dim=0)

    causal_masks = []
    for q_seq_len, k_seq_len in zip(q_seq_lens, k_seq_lens):
        mask = torch.ones(q_seq_len, k_seq_len, dtype=torch.bool, device=q.device)
        mask = mask.tril(diagonal=k_seq_len - q_seq_len)
        causal_masks.append(mask.reshape(-1))
    causal_masks = torch.cat(causal_masks, dim=0)

    streamingllm_masks = []
    attention_sink = 1
    window_size = 3
    for i, (q_seq_len, k_seq_len) in enumerate(zip(q_seq_lens, k_seq_lens)):
        mask = torch.ones(size=(q_seq_len, k_seq_len), dtype=torch.bool, device=device)
        mask = torch.triu(mask, diagonal=k_seq_len - q_seq_len - window_size)
        mask = torch.tril(mask, diagonal=k_seq_len - q_seq_len)
        mask[:, :attention_sink] = 1

        mask = torch.randint(0, 2, (q_seq_len, k_seq_len), dtype=torch.bool, device=device)
        streamingllm_masks.append(mask.reshape(-1))
    streamingllm_masks = torch.cat(streamingllm_masks, dim=0)

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    from hydrainfer.utils.profiler import profile


    with profile('full mask'):
        for i in range(100):
            prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, "NHD", use_cuda_graph=False)
            prefill_wrapper.plan(
                qo_indptr              = qo_indptr,
                paged_kv_indptr        = paged_kv_indptr,
                paged_kv_indices       = paged_kv_indices,
                paged_kv_last_page_len = paged_kv_last_page_len,
                num_qo_heads           = num_qo_heads,
                num_kv_heads           = num_kv_heads,
                head_dim               = head_dim,
                page_size              = page_size,
                # causal                 = True,
                custom_mask            = full_masks, 
            )
            o = prefill_wrapper.run(q, (k_cache, v_cache))
    
    with profile('causal mask'):
        for i in range(100):
            prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, "NHD", use_cuda_graph=False)
            prefill_wrapper.plan(
                qo_indptr              = qo_indptr,
                paged_kv_indptr        = paged_kv_indptr,
                paged_kv_indices       = paged_kv_indices,
                paged_kv_last_page_len = paged_kv_last_page_len,
                num_qo_heads           = num_qo_heads,
                num_kv_heads           = num_kv_heads,
                head_dim               = head_dim,
                page_size              = page_size,
                # causal                 = True,
                custom_mask            = causal_masks, 
            )
            o = prefill_wrapper.run(q, (k_cache, v_cache))
    
    with profile('streamingllm_mask'):
        for i in range(100):
            prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, "NHD", use_cuda_graph=False)
            prefill_wrapper.plan(
                qo_indptr              = qo_indptr,
                paged_kv_indptr        = paged_kv_indptr,
                paged_kv_indices       = paged_kv_indices,
                paged_kv_last_page_len = paged_kv_last_page_len,
                num_qo_heads           = num_qo_heads,
                num_kv_heads           = num_kv_heads,
                head_dim               = head_dim,
                page_size              = page_size,
                # causal                 = True,
                custom_mask            = streamingllm_masks, 
            )
            o = prefill_wrapper.run(q, (k_cache, v_cache))
    # conclusion: attention mask will not reduce masked dot product computation

def test_attention_param_creation_perf():
    n_layers = 32
    batch_size = 1
    n_tokens = 200
    k_seq_lens = [[random.randint(0, 10) for _ in range(batch_size)] for _ in range(n_layers)]
    
    device = torch.device('cuda:0')
    from hydrainfer.utils.profiler import profile
    with profile('many attention_param creation'):
        for _ in range(n_tokens):
            for layer_id in range(n_layers):
                k_cu_seq_lens = torch.tensor(list(accumulate(k_seq_lens[layer_id], initial=0)), dtype=torch.int, device=device)

    with profile('one attention_param creation'):
        for _ in range(n_tokens):
            k_cu_seq_lens = torch.tensor([list(accumulate(k_seq_lens[layer_id], initial=0)) for _ in range(n_layers)], dtype=torch.int, device=device)
            for layer_id in range(n_layers):
                layer_k_cu_seq_lens = k_cu_seq_lens[layer_id]
    # conclusion: create attention param in one creation and use slice to partition data will save time

if __name__ == '__main__':
    pytest.main([__file__ + '::test_flash_attn_func', '-x'])
    pytest.main([__file__ + '::test_flash_infer_batch_decode', '-x'])
    pytest.main([__file__ + '::test_flash_infer_batch_prefill', '-x'])
    pytest.main([__file__ + '::test_flash_infer_batch_prefill_mask', '-x'])
    # test_flash_infer_batch_prefill_mask_perf()
    # test_attention_param_creation_perf()