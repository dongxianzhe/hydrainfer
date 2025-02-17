import torch
from torch import Tensor

def set_image_cache(
    slot_ids: Tensor,          # [n_tokens]
    image_tokens: Tensor,      # [n_tokens, n_heads, head_dim]
    image_cache: Tensor):      # [n_blocks, block_size, n_heads, head_dim]
    ...