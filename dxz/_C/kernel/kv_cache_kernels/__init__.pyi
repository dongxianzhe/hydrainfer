import torch
from torch import Tensor

# note that pybind module can't be called by key value form
def set_kv_cache(
    slot_ids: Tensor,  # [n_tokens]
    keys: Tensor,      # [n_tokens, n_kv_heads, head_dim]
    values: Tensor,    # [n_tokens, n_kv_heads, head_dim]
    key_cache: Tensor,  # [n_blocks, block_size, n_heads, head_dim]
    value_cache: Tensor) -> None:
    ...