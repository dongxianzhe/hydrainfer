import torch
from torch import Tensor

# apply rotary embedding to query and key inplace
def apply_rotary_pos_emb(
    query: Tensor,            # [n_tokens, n_heads, head_dim]
    key: Tensor,              # [n_tokens, n_kv_heads, head_dim]
    positions: Tensor,        # [n_tokens]
    cos_sin: Tensor,          # [max_positions, 2, rotary_dim/2]
    rotary_dim: int,
    interleaved: bool) -> None:
    ...