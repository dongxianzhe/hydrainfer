import math
import torch
import flashinfer
from dataclasses import dataclass
from torch import Tensor
import pytest

from hydrainfer.model.parameters import AttentionParameters
from torch import nn
@dataclass
class MultiHeadLatentAttentionConfig:
    hidden_size: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    n_heads: int
    max_position_embeddings: int
    theta: float

from hydrainfer.layer.rotary_embedding import RotaryEmbedding, compute_default_inv_freq

class MultiHeadLatentAttention:
    def __init__(self, config: MultiHeadLatentAttentionConfig) -> None:
        self.w1 = nn.Linear(config.hidden_size, config.q_lora_rank)
        self.w2 = nn.Linear(config.q_lora_rank, config.n_heads * config.qk_nope_head_dim)
        self.w3 = nn.Linear(config.q_lora_rank, config.n_heads * config.qk_rope_head_dim)
        self.w4 = nn.Linear(config.hidden_size, config.kv_lora_rank)
        self.w5 = nn.Linear(config.kv_lora_rank, config.qk_rope_head_dim)
        self.w6 = nn.Linear(config.kv_lora_rank, config.n_heads * config.qk_nope_head_dim)
        self.w7 = nn.Linear(config.kv_lora_rank, config.n_heads * config.v_head_dim)
        self.o_proj = nn.Linear(config.n_heads * config.v_head_dim, config.hidden_size)
        self.rope = RotaryEmbedding(
            rotary_dim = config.qk_rope_head_dim, 
            max_position_embeddings = config.max_position_embeddings, 
            inv_freq = compute_default_inv_freq(
                rotary_dim = config.qk_rope_head_dim, 
                theta = config.theta, 
            ), 
            interleaved = False, 
        )

    def forward(self, h: Tensor, position_ids: Tensor, attention_params: AttentionParameters) -> Tensor:
        c_q = self.w1(h)
        q_c = self.w2(h)
        q_r = self.w3(c_q)
        pass

# @pytest.mark.parametrize("size", [(4096, 4096)])
# @pytest.mark.parametrize("device", [torch.device('cuda:0')])
# def test_slice_perf(
#     size: tuple[int, int], 
#     device: torch.device, 
# ):
#     t = torch.randn(size=size, device=device)
#     ts = []
#     def slice_fun():
#         t1 = t[:, :2048]
#         t2 = t[:, 2048:]
#         ts.append(t1)
#         ts.append(t2)
#         return t1, t2

#     def test_copy():
#         z = t.clone()
#         ts.append(z)
#         return z

#     from hydrainfer.utils.benchmark_utils import benchmark
#     latency1 = benchmark(slice_fun)
#     latency2 = benchmark(test_copy)
#     print(f'benchmark(slice_fun) {latency1} s')
#     print(f'benchmark(test_copy) {latency2} s')
#     print(f'ratio {latency1 / latency2}')

@pytest.mark.parametrize("seq_lens", [[(1, 100), (15, 15), (111, 234), (2, 1024)]])
# @pytest.mark.parametrize("num_heads", [8, 4, 2, 1])
@pytest.mark.parametrize("num_heads", [8])
@pytest.mark.parametrize("head_size", [(128, 64, 128)])
@pytest.mark.parametrize("kv_lora_rank", [128])
@pytest.mark.parametrize("n_blocks", [10000])
@pytest.mark.parametrize("block_size", [16])
# @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("device", [torch.device('cuda:0')])
@torch.inference_mode
def test_mla_causal_attention(
    seq_lens: list[tuple[int, int]],
    num_heads: int,
    head_size: tuple[int, int, int],
    n_blocks: int,
    block_size: int,
    kv_lora_rank: int, 
    dtype: torch.dtype,
    device: torch.device
):
    batch_size = len(seq_lens)
    qk_nope_head_dim, qk_rope_head_dim, v_head_dim = head_size

    head_dim_ckv = kv_lora_rank # todo
    head_dim_kpe = qk_rope_head_dim # todo
    page_size = block_size # todo


    q_seq_lens: list[int] = []
    kv_seq_lens: list[int] = []
    block_lens: list[int] = []
    for q_seq_len, kv_seq_len in seq_lens:
        q_seq_lens.append(q_seq_len)
        kv_seq_lens.append(kv_seq_len)
        block_lens.append((kv_seq_len + block_size - 1) / block_size)
    from itertools import accumulate
    q_cu_seq_lens: list[int] = list(accumulate(q_seq_lens, initial=0))
    kv_cu_seq_lens: list[int] = list(accumulate(kv_seq_lens, initial=0))
    block_tables: list[int] = list(range(len(block_lens)))

    block_tables_t: Tensor = torch.tensor(block_tables, dtype=torch.int, device=device)
    q_cu_seq_lens_t: Tensor = torch.tensor(q_cu_seq_lens, dtype=torch.int, device=device)
    kv_seq_lens_t: Tensor = torch.tensor(kv_seq_lens, dtype=torch.int, device=device)
    kv_cu_seq_lens_t: Tensor = torch.tensor(kv_cu_seq_lens, dtype=torch.int, device=device)

    print(q_cu_seq_lens)
    kv_cache = torch.randn(size=(n_blocks, block_size, 1, kv_lora_rank + qk_rope_head_dim), dtype=dtype, device=device)

    q_nope = torch.randn(size=(q_cu_seq_lens[-1], num_heads, qk_nope_head_dim), dtype=dtype, device=device)
    q_pe = torch.randn(size=(q_cu_seq_lens[-1], num_heads, qk_rope_head_dim), dtype=dtype, device=device)
    print(f'q_nope.shape {q_nope.shape}')
    cache = torch.randn(size=(n_blocks, block_size, 1, kv_lora_rank + qk_rope_head_dim))
    c_kv = cache[:, :, :, :kv_lora_rank].view(n_blocks, block_size, kv_lora_rank)
    k_pe = cache[:, :, :, kv_lora_rank:].view(n_blocks, block_size, qk_rope_head_dim)
    sm_scale = 1.0 / (math.sqrt(qk_nope_head_dim + qk_rope_head_dim))  # use head dimension before matrix absorption
    mla_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device), 
        backend="fa2"
    )
    mla_wrapper.plan(
        qo_indptr = q_cu_seq_lens_t, 
        kv_indptr = kv_cu_seq_lens_t, 
        kv_indices = block_tables_t, 
        kv_len_arr = kv_seq_lens_t, 
        num_heads = num_heads,
        head_dim_ckv = kv_lora_rank, 
        head_dim_kpe = qk_rope_head_dim,
        page_size = block_size,
        causal = True,
        sm_scale = sm_scale,
        q_data_type = dtype, 
        kv_data_type = dtype, 
        use_profiler = False, 
    )
    o = mla_wrapper.run(q_nope, q_pe, c_kv, k_pe, return_lse=False)
    print(f'o.shape {o.shape}')

if __name__ == '__main__':
    pytest.main([__file__, '-s', '-x'])