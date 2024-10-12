# void mha_varlen_fwd(at::Tensor& out,             // [n_tokens, n_heads, head_dim]
#                const at::Tensor& q,         // [n_tokens, n_heads, head_dim]
#                const at::Tensor& k,
#                const at::Tensor& v,
#                const at::Tensor& cu_seqlens_q,  // [batch + 1]
#                const at::Tensor& cu_seqlens_k,  // [batch + 1]
#                const c10::optional<at::Tensor>& block_table_, // [batch, max_blocks_per_seq]
#                const c10::optional<at::Tensor>& cu_block_lens, // [batch + 1]
#                const c10::optional<at::Tensor>& alibi_slopes, // [num_heads]
#                int max_seqlen_q,      // max sequence length for Q
#                int max_seqlen_k,      // max sequence length for K/V
#                float softmax_scale,
#                float softcap,
#                int window_size_left,
#                int window_size_right,
#                int num_splits);

import torch
from torch import Tensor
from typing import Optional

# note that pybind module can't be called by key value form
def mha_varlen_fwd(
    out: Tensor,             # [n_tokens, n_heads, head_dim]
    q: Tensor,         # [n_tokens, n_heads, head_dim]
    k: Tensor,
    v: Tensor,
    cu_seqlens_q: Tensor,  # [batch + 1]
    cu_seqlens_k: Tensor,  # [batch + 1]
    block_table_: Optional[Tensor], # [batch, max_blocks_per_seq]
    cu_block_lens: Optional[Tensor], # [batch + 1]
    alibi_slopes: Optional[Tensor], # [num_heads]
    max_seqlen_q: int,      # max sequence length for Q
    max_seqlen_k: int,      # max sequence length for K/V
    softmax_scale: float,
    softcap: float,
    window_size_left: int,
    window_size_right: int,
    num_splits: int
):
    ...