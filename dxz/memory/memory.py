import torch
from torch import Tensor

try:
    from dxz._C.kernel.kv_cache_kernels import set_kv_cache as set_kv_cache_kernel
except:
    print('import set_kv_cache failed')
    set_kv_cache_kernel = None


class KVCache:
    """
        one layer's kv_cache
    """
    def __init__(self, kv_cache: Tensor):
        assert kv_cache.shape.dim() == 5, f"kv_cache dim should be (2, n_blocks, block_size, num_heads, head_size) but got {kv_cache.shape}"
        assert kv_cache.shape[0] == 2, f"kv_cache first dim should be 2 but got {kv_cache.shape}"
        self.key_cache = kv_cache[0]
        self.value_cache = kv_cache[1]

    def get_kv_cache(self) -> tuple[Tensor, Tensor]: 
        return self.key_cache, self.value_cache

    def set_kv_cache(self, slot_ids: Tensor, keys: Tensor, values: Tensor):
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