import torch
from torch import Tensor

try:
    from dxz._C.kernel.kv_cache_kernels import set_kv_cache as set_kv_cache_kernel
except:
    print('import set_kv_cache failed')
    set_kv_cache_kernel = None

class KVCache:
    # todo pass in kv shape and option instead
    def __init__(self, key_cache: Tensor, value_cache: Tensor):
        # key_cache   (num_blocks, block_size, num_kv_heads, head_size)
        # value_cache (num_blocks, block_size, num_kv_heads, head_size)
        assert key_cache.shape == value_cache.shape
        assert key_cache.device == value_cache.device
        assert key_cache.dim() == 4

        self.key_cache = key_cache
        self.value_cache = value_cache
        self.num_blocks, self.block_size, self.num_kv_heads, self.head_size = key_cache.shape
        self.device = key_cache.device
    
    def get_kv_cache(self) -> tuple[Tensor, Tensor]:
        return (self.key_cache, self.value_cache)
    
    def set_kv_cache(self, slot_ids: Tensor, keys: Tensor, values: Tensor) -> None:
        # set key tokens and value tokens to kv cache, each slot store one token
        # slot_ids (num_slots, ) int
        # keys (num_slots, num_heads, head_dim)
        # values (num_slots, num_heads, head_dim)
        assert slot_ids.shape[0] == keys.shape[0]
        assert slot_ids.shape[0] == values.shape[0]
        assert slot_ids.device == keys.device
        assert slot_ids.device == values.device
        if self.device.type == 'cuda':
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