import torch
from torch import Tensor
from dxz.memory.block_allocator import BlockAllocator
from dxz.memory import TokenCache

try:
    from dxz._C.kernel.kv_cache_kernels import set_kv_cache as set_kv_cache_kernel
except:
    print('import set_kv_cache failed')
    set_kv_cache_kernel = None

class KVCache:
    def __init__(self, key_cache: Tensor, value_cache: Tensor):
        assert key_cache.dim() == 4, f'key cache dim should be 4 but got shape {key_cache.shape}'
        assert value_cache.shape == key_cache.shape, f'key cache shape should be same with value cache but got {key_cache.shape} {value_cache.shape}'
        self.key_cache = key_cache
        self.value_cache = value_cache
        self.dtype = key_cache.dtype
        self.device = key_cache.device

    def get_kv_cache(self) -> tuple[Tensor, Tensor]:
        return (self.key_cache, self.value_cache)
    
    def set_kv_cache(self, slot_ids: Tensor, keys: Tensor, values: Tensor) -> None:
        # set key tokens and value tokens to kv cache, each slot store one token
        # slot_ids (num_slots, ) int
        # keys (num_slots, num_heads, head_dim)
        # values (num_slots, num_heads, head_dim)
        assert slot_ids.shape[0] == keys.shape[0], f"{slot_ids.shape} {keys.shape}"
        assert slot_ids.shape[0] == values.shape[0], f"{slot_ids.shape} {values.shape}"
        assert slot_ids.device == keys.device, f"{slot_ids.device} {keys.device}"
        assert slot_ids.device == values.device, f"{slot_ids.device} {values.device}"
        if self.device.type == 'cuda' and set_kv_cache_kernel:
            set_kv_cache_kernel(
                slot_ids,
                keys,
                values,
                self.key_cache,
                self.value_cache,
            )
        else:
            num_tokens = slot_ids.shape[0]
            for i in range(num_tokens):
                block_id = slot_ids[i] // self.block_size
                block_offset = slot_ids[i] % self.block_size
                self.key_cache[block_id, block_offset, :, :] = keys[i, :, :]
                self.value_cache[block_id, block_offset, :, :] = values[i, :, :]

    @classmethod
    def from_token_cache(cls, token_cache: TokenCache):
        tensors = TokenCache.get_caches
        assert len(tensors) == 2, f'can not convert token_cache with {len(tensors)} to kv cache'
        return KVCache(token_cache[0], token_cache[1])