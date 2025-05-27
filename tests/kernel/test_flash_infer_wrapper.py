from typing import List, Tuple

import pytest
import torch

# from hydrainfer._C.kernel.flash_infer import BatchPrefillWrapper
# from hydrainfer._C.kernel import flash_infer
import hydrainfer._C.kernel.flash_infer as flash_infer
# import scalellm._C.kernels as kernels

from typing import List, Optional

import torch

def masked_self_attention(
    query: torch.Tensor,  # [q_len, n_heads, head_dim]
    key: torch.Tensor,  # [kv_len, n_heads, head_dim]
    value: torch.Tensor,  # [kv_len, n_heads, head_dim]
    alibi_bias: Optional[torch.Tensor], # [n_heads, 1, kv_len]
    mask: torch.Tensor,  # [n_heads, q_len, kv_len]
    sm_scale: float,
    logits_soft_cap: float,
) -> torch.Tensor:
    # => [n_heads, q_len, kv_len]
    scores = torch.einsum("qhd,khd->hqk", query.float(), key.float())

    # apply scale
    scores *= sm_scale

    # apply soft_cap
    if logits_soft_cap > 0.0:
        scores = torch.tanh(scores / logits_soft_cap) * logits_soft_cap
        
    # apply alibi bias
    if alibi_bias is not None:
        scores += alibi_bias

    # apply mask
    scores.masked_fill_(mask == 0, float("-inf"))

    # softmax => [n_heads, q_len, kv_len]
    scores = torch.softmax(scores, dim=-1)
    # => [q_len, n_heads, head_dim]
    return torch.einsum("hqk,khd->qhd", scores, value.float()).type_as(query)


def varlen_masked_self_attention(
    query: torch.Tensor,  # [q_len, n_heads, head_dim]
    key_cache: torch.Tensor,  # [n_blocks, block_size, n_kv_heads, head_dim]
    value_cache: torch.Tensor,  # [n_blocks, block_size, n_kv_heads, head_dim]
    query_lens: List[int],  # [batch_size]
    kv_lens: List[int],  # [batch_size]
    block_tables: torch.Tensor,  # block ids for each sequence, [batch_size, max_num_blocks]
    sm_scale: float,
    logits_soft_cap: float = 0.0,
    sliding_window: int = -1,
    alibi_slopes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert key_cache.shape == value_cache.shape

    n_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, n_heads, head_dim = query.shape
    _, block_size, n_kv_heads, head_size = key_cache.shape

    outputs: List[torch.Tensor] = []
    start_idx = 0
    # process sequence one by one
    for i in range(n_seqs):
        q_len = query_lens[i]
        kv_len = kv_lens[i]
        assert kv_len >= q_len

        # [q_len, n_heads, head_dim]
        q = query[start_idx : start_idx + q_len]

        # retrieve key and value from cache
        num_kv_blocks = (kv_len + block_size - 1) // block_size
        # block ids for current sequence
        block_ids = block_tables[i, :num_kv_blocks]

        # [kv_len, n_kv_heads, head_size]
        k = key_cache[block_ids].view(-1, n_kv_heads, head_size)[:kv_len]
        v = value_cache[block_ids].view(-1, n_kv_heads, head_size)[:kv_len]

        if n_heads != n_kv_heads:
            assert n_heads % n_kv_heads == 0
            n_groups = n_heads // n_kv_heads
            k = k.repeat_interleave(repeats=n_groups, dim=1)
            v = v.repeat_interleave(repeats=n_groups, dim=1)

        # create mask [1, q_len, kv_len]
        mask = torch.ones(1, q_len, kv_len).bool()
        if sliding_window >= 0:
            # returns the upper triangular part of a matrix
            mask = torch.triu(mask, diagonal=kv_len - q_len - sliding_window)

        # causal mask
        # returns the lower triangular part of a matrix
        mask = mask.tril(diagonal=kv_len - q_len).to(query)

        # calculate alibi attention bias
        alibi_bias = None
        if alibi_slopes is not None:
            assert alibi_slopes.shape == (n_heads,)
            # since it's causal mask, we can just use [0, 1, ...,, kv_len)
            distance = torch.arange(kv_len, dtype=torch.float32)
            # [n_heads, 1, kv_len]
            alibi_bias = distance.view(1, 1, -1) * alibi_slopes.view(n_heads, 1, 1)
            

        out = masked_self_attention(
            query=q,
            key=k,
            value=v,
            alibi_bias=alibi_bias,
            mask=mask,
            sm_scale=sm_scale,
            logits_soft_cap=logits_soft_cap,
        )

        outputs.append(out)
        start_idx += q_len

    return torch.cat(outputs, dim=0)

@pytest.mark.parametrize("seq_lens", [[(1, 100), (15, 15), (111, 234), (1000, 10000)]])
@pytest.mark.parametrize("num_heads", [(8, 8), (8, 4), (8, 2), (8, 1)])
@pytest.mark.parametrize("head_size", [64, 128, 256])
@pytest.mark.parametrize("n_blocks", [100])
@pytest.mark.parametrize("block_size", [4, 8, 16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 50.0])
@pytest.mark.parametrize("sliding_window", [-1, 50])
@pytest.mark.parametrize("alibi", [False, True])
@torch.inference_mode
def test_flashinfer_varlen_masked_self_attention(
    seq_lens: List[Tuple[int, int]],
    num_heads: Tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    n_blocks: int,
    block_size: int,
    logits_soft_cap: float,
    sliding_window: int,
    alibi: bool,
) -> None:
    torch.set_default_device("cuda")
    torch.cuda.manual_seed_all(0)

    n_seqs = len(seq_lens)
    q_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]

    n_heads, n_kv_heads = num_heads
    assert n_heads % n_kv_heads == 0
    max_kv_len = max(kv_lens)
    sm_scale = head_size**-0.5

    # Generate random query, key, and value tensors.
    query = torch.randn(sum(q_lens), n_heads, head_size, dtype=dtype)
    key_cache = torch.randn(n_blocks, block_size, n_kv_heads, head_size, dtype=dtype)
    value_cache = torch.randn(n_blocks, block_size, n_kv_heads, head_size, dtype=dtype)

    max_n_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, n_blocks, (n_seqs, max_n_blocks_per_seq), dtype=torch.int32
    )

    # prepare input tensors for the kernel
    qo_indptr = [0]
    kv_indptr = [0]
    paged_kv_indptr = [0]
    paged_kv_indices = []
    for i in range(n_seqs):
        qo_indptr.append(qo_indptr[-1] + q_lens[i])
        kv_indptr.append(kv_indptr[-1] + kv_lens[i])

        seq_len = kv_lens[i]
        assert seq_len > 0

        num_blocks = (seq_len + block_size - 1) // block_size
        paged_kv_indices.extend(block_tables[i, :num_blocks])
        paged_kv_indptr.append(len(paged_kv_indices))

    qo_indptr = torch.tensor(qo_indptr, dtype=torch.int32)
    kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32)
    paged_kv_indptr = torch.tensor(paged_kv_indptr, dtype=torch.int32)
    paged_kv_indices = torch.tensor(paged_kv_indices, dtype=torch.int32)

    wrapper = flash_infer.BatchPrefillWrapper(False)
    # wrapper = kernels.BatchPrefillWrapper(False)
    # TODO: determine the best size for the workspace buffer.
    float_workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8)
    int_workspace_buffer = torch.empty(8 * 1024 * 1024, dtype=torch.uint8)

    empty_q_data = torch.empty(0, dtype=dtype)

    wrapper.plan(
        float_workspace_buffer,
        int_workspace_buffer,
        qo_indptr,
        paged_kv_indptr,
        n_seqs,
        n_heads,
        n_kv_heads,
        head_size,
        block_size,
        empty_q_data,
    )

    alibi_slopes = torch.randn(n_heads, dtype=torch.float32) if alibi else None

    output = wrapper.run(
        query,
        qo_indptr,
        kv_indptr,
        key_cache,
        value_cache,
        paged_kv_indptr,
        paged_kv_indices,
        sliding_window,
        logits_soft_cap,
        sm_scale,
        alibi_slopes,
    )

    ref_output = varlen_masked_self_attention(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=q_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        sm_scale=sm_scale,
        logits_soft_cap=logits_soft_cap,
        sliding_window=sliding_window,
        alibi_slopes=alibi_slopes,
    )

    if alibi and dtype == torch.bfloat16:
        torch.testing.assert_close(output, ref_output, atol=2e-2, rtol=1e-2)
    else:
        torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
