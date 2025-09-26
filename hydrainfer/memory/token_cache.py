import torch
from torch import Tensor
from dataclasses import dataclass, field
from typing import Optional, Literal
from hydrainfer.utils.logger import getLogger
logger = getLogger(__name__)

try:
    from hydrainfer._C.kernel.cache_kernels import set_image_cache
except ImportError:
    set_image_cache = None
    logger.warning('set_image_cache import failed')


class TokenCache:
    """
    # for key value cache the caches should be [key_cache, value_cache]
    #     key_cache has shape (n_blocks, block_size, n_kv_heads, head_size)
    #     value_cache has shape (n_blocks, block_size, n_kv_heads, head_size)
    for image_embed_cache the caches should be [image_embed_cache]
        image_embed_cache has shape(n_blocks, block_size, n_qo_heads, head_size)
    """
    def __init__(self, caches: list[Tensor]):
        for cache in caches:
            assert cache.dim() == 4, f'cache dim should be 4 (n_blocks, block_size, n_kv_heads, head_size) but got shape {cache.shape}'
            assert cache.shape == caches[0].shape, f'all shape should be same but got {cache.shape} and {caches[0].shape}'
            assert cache.dtype == caches[0].dtype, f'all dtype should be same but got {cache.dtype} and {caches[0].dtype}'
            assert cache.device == caches[0].device, f'all device should be same but got {cache.device} and {caches[0].device}'
        self.caches = caches
        self.block_size = caches[0].shape[1]
        self.dtype = caches[0].dtype
        self.device = caches[0].device

    def get_caches(self) -> list[Tensor]:
        return self.caches

    def set_caches(self, slot_ids: Tensor, values: list[Tensor]):
        assert slot_ids.dim() == 1
        for value in values:
            assert value.dim() == 3
            assert slot_ids.shape[0] == value.shape[0], f"{slot_ids.shape[0]} {value.shape[0]}"
            assert slot_ids.device == value.device, f"{slot_ids.device} {value.device}"

        if self.device.type == 'cuda' and set_image_cache:
            for cache, value in zip(self.caches, values):
                set_image_cache(
                    slot_ids, 
                    value,  
                    cache,
                )
            return

        for cache, value in zip(self.caches, values):
            slot_view = cache.view(-1, cache.shape[-2], cache.shape[-1])
            slot_view[slot_ids, :, :] = value
            return


@dataclass
class VirtualTokenCache:
    vid: int
    n_blocks_of_cache_manager: int # used to compute block tensor's index when migrating cache
    n_cache_tokens: int = 0
    block_table: list[int] = field(default_factory=list)
    memory_handle: Optional[list[int]] = None # used in cuda ipc memory handle
    rank: int = -1 # used in nccl

