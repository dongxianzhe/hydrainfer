import torch
from torch import Tensor

try:
    from dxz._C.kernel.kv_cache_kernels import set_kv_cache as set_kv_cache_kernel
except:
    print('import set_kv_cache failed')
    set_kv_cache_kernel = None

class KVCache:
    def __init__(self, num_blocks: int, block_size: int, num_kv_heads: int, head_size: int, dtype: torch.dtype, device: torch.device):
        self.num_blocks   = num_blocks
        self.block_size   = block_size
        self.num_kv_heads = num_kv_heads
        self.head_size    = head_size
        self.dtype        = dtype
        self.device       = device
        self.key_cache   = torch.randn(size=(num_blocks, block_size, num_kv_heads, head_size), dtype=dtype, device=device)
        self.value_cache = torch.randn(size=(num_blocks, block_size, num_kv_heads, head_size), dtype=dtype, device=device)
    
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