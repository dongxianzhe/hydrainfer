import torch
from torch import Tensor
from dataclasses import dataclass, field
from typing import Optional
from dxz._C.data_transfer.block_migration import get_ipc_mem_handle
from dxz._C.data_transfer import block_migration
from dxz.memory import BlockAllocator
from dxz.utils.allocate import IncreaingAllocator


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
        self.dtype = caches[0].dtype
        self.device = caches[0].device

    def get_caches(self) -> list[Tensor]:
        return self.caches

    def set_caches(self, slot_ids: Tensor, values: list[Tensor]):
        assert slot_ids.dim() == 1
        for value in values:
            assert value.dim() == 3
            assert slot_ids.shape[0] == value.shape[0], f"{slot_ids[0].shape} {value.shape[0]}"
            assert slot_ids.device == value.device, f"{slot_ids.device} {value.device}"

        for cache, value in zip(self.caches, values):
            slot_view = cache.view(-1, cache.shape[-2], cache.shape[-1])
            slot_view[slot_ids, :, :] = value


@dataclass
class VirtualTokenCache:
    vid: int
    n_cache_tokens: int = 0
    block_table: list[int] = field(default_factory=list)
    memory_handle: Optional[list[int]] = None


@dataclass
class TokenCacheBlockManagerContext:
    n_layers: int
    n_tokens: int
    n_blocks: int
    block_size: int
    n_heads: int
    head_size: int
    dtype: torch.dtype
    device: torch.device


class TokenCacheBlockManager:
    def __init__(self, context: TokenCacheBlockManagerContext):
        self.n_layers   = context.n_layers
        self.n_tokens   = context.n_tokens 
        self.n_blocks   = context.n_blocks
        self.block_size = context.block_size
        self.n_heads    = context.n_heads
        self.head_size  = context.head_size
        self.dtype      = context.dtype
        self.device     = context.device

        self.cache_tensor = torch.randn(size=(self.n_layers, self.n_tokens, self.n_blocks, self.block_size, self.n_heads, self.head_size), dtype=self.dtype, device=self.device)
        self.memory_handle: list[int] = get_ipc_mem_handle(self.cache_tensor)
        self.block_allocator = BlockAllocator(self.n_blocks)
        self.vid_allocator = IncreaingAllocator(first_value=1)

    def allocate_virtual_cache(self) -> VirtualTokenCache:
        return VirtualTokenCache(
            vid = self.vid_allocator.allocate(), 
            n_cache_tokens = 0, 
            block_table = [], 
            memory_handle = self.memory_handle,
        )

    def v2p(self, virtual_cache: VirtualTokenCache, virtual_cache_ids: list[int]) -> list[int]:
        physical_cache_ids: list[int] = []
        for vcid in virtual_cache_ids:
            block_id = vcid // self.block_size
            block_offset = vcid % self.block_size
            slot_id = virtual_cache.block_table[block_id] * self.block_size + block_offset
            physical_cache_ids.append(slot_id)
        return physical_cache_ids

    def set(self, virtual_cache: VirtualTokenCache, virtual_cache_ids: list[int]) -> list[int]:
        # 1. try to allocate memory if block is not enough
        n_tokens = max(virtual_cache_ids) + 1
        n_blocks = (n_tokens + self.block_size - 1) // self.block_size
        if len(virtual_cache.block_table) < n_blocks:
            virtual_cache.block_table += self.block_allocator.allocate(n_blocks - len(virtual_cache.block_table))
        if len(virtual_cache.block_table) < n_blocks:
            raise Exception(f'not enough cache, total n_blocks {self.n_blocks}')
        # 2. set vitual cache
        virtual_cache.n_cache_tokens = max(virtual_cache.n_cache_tokens, n_tokens)

        # 3. get phyical cache slot id
        return self.v2p(virtual_cache, virtual_cache_ids)

    def free_blocks(self, virtual_cache: VirtualTokenCache, virtual_block_ids: list[int]):
        for virtual_block_id in sorted(virtual_block_ids, reverse=True):
            physical_block_id = virtual_cache.block_table[virtual_block_id]
            self.block_allocator.free([physical_block_id])
            if virtual_block_id == len(virtual_cache.block_tables) - 1:
                virtual_cache.n_cache_tokens -= (virtual_cache.n_cache_tokens + self.block_size - 1) % self.block_size + 1
            else:
                virtual_cache.n_cache_tokens -= self.block_size
            del virtual_cache.block_table[virtual_block_id]

    def realloc(self, virtual_cache: VirtualTokenCache, n_tokens: int):
        if n_tokens > virtual_cache.n_cache_tokens:
            n_need_blocks = (n_tokens + self.block_size - 1) // self.block_size
            virtual_cache.block_table += self.block_allocator.allocate(n_need_blocks - len(virtual_cache.block_table))
            virtual_cache.n_cache_tokens = n_tokens
        else:
            n_need_blocks = (n_tokens + self.block_size - 1) // self.block_size
            self.block_allocator.free(virtual_cache.block_table[n_need_blocks:])
            virtual_cache.block_table = virtual_cache.block_table[:n_need_blocks]
            virtual_cache.n_cache_tokens = n_tokens

    def get_layer_cache(self, layer_id: int) -> TokenCache:
        return TokenCache([self.cache_tensor[layer_id, token_id, :, :, :, :] for token_id in range(self.n_tokens)])

    def migrate_blocks(self, src_virtual_cache: VirtualTokenCache, dst_virtual_cache: VirtualTokenCache):
        assert src_virtual_cache.n_cache_tokens == dst_virtual_cache.n_cache_tokens, f'{src_virtual_cache.n_cache_tokens} {dst_virtual_cache.n_cache_tokens}'
        # if src_memory_handle not in self.src_memory_handle_dict:
        #     block_migration.register_ipc_mem_handle(src_memory_handle)
        #     src_memory_handle_dict[]
        assert src_virtual_cache.memory_handle is not None
        dev_ptr = block_migration.register_ipc_mem_handle(src_virtual_cache.memory_handle)
        # todo modify cpp code to support general layers migrate
        # todo memory copy stream
        for layer_id in range(self.n_layers):
            block_migration.migrate_blocks(
                0, # prefill_start_head: int
                self.n_heads, # prefill_end_head: int
                src_virtual_cache.block_table, # prefill_block_indexes: list[int]
                0, # decoding_start_head: int
                self.n_heads, # decoding_end_head:int
                dst_virtual_cache.block_table, # decoding_block_indexes: list[int]
                dev_ptr, # prefill_dev_ptr_index: int
                self.n_heads, # num_heads: int
                self.cache_tensor[layer_id, :, :, :, :, :], # decoding_worker_kv_cache: Tensor
            )
        torch.cuda.synchronize()
        print('migrate_blocks cuda synchronize')