import torch
from dxz.memory.block_allocator import BlockAllocator
from dxz.memory.kv_cache import KVCache
from dxz.memory.virtual_kv_cache import VirtualKVCache
from dxz.memory.memory_management import MemoryConfig, MemoryContext, MemoryManagementUnit


class SharedMemoryManagementUnit(MemoryManagementUnit):
    """
    all layer kv cache shared one kv cache pool
    """
    def __init__(self, config: MemoryConfig, context: MemoryContext):
        self.config = config
        self.context = context
        assert self.config.memory_management_policy == 'shared'
        self.kv_cache = torch.randn(size=(2, config.num_blocks, config.block_size, context.num_kv_heads, context.head_size), dtype=context.dtype, device=context.device)
        self.kv_cache_pool = KVCache(self.kv_cache[0, :, :, :, :], self.kv_cache[1, :, :, :, :])
        self.block_allocator = BlockAllocator(config.num_blocks)

        self.vid_allocator = 0
        self.vid2kv_cache = {}


    def allocate_virtual_kv_caches(self, n_virtual_kv_caches: int) -> list[VirtualKVCache]:
        assert n_virtual_kv_caches == self.context.n_layers, "not supported other allocation yet"
        virtual_kv_caches: list["VirtualKVCache"] = []
        for layer_id in range(n_virtual_kv_caches):
            vid = self.vid_allocator
            self.vid_allocator += 1
            virtual_kv_cache = VirtualKVCache(
                vid = vid, 
                n_kv_cache_tokens = 0, 
                block_table = [], 
            )
            virtual_kv_caches.append(virtual_kv_cache)
            self.vid2kv_cache[vid] = self.kv_cache_pool
        return virtual_kv_caches


    def v2p(self, virtual_kv_cache: VirtualKVCache, virtual_cache_ids: list[int]) -> list[int]:
        slot_ids: list[int] = []
        for vcid in virtual_cache_ids:
            block_id = vcid // self.config.block_size
            block_offset = vcid % self.config.block_size
            slot_id = virtual_kv_cache.block_table[block_id] * self.config.block_size + block_offset
            slot_ids.append(slot_id)
        return slot_ids

    def set(self, virtual_kv_cache: VirtualKVCache, virtual_cache_ids: list[int]) -> list[int]:
        # 1. try to allocate memory if block is not enough
        n_tokens = max(virtual_cache_ids) + 1
        n_blocks = (n_tokens + self.config.block_size - 1) // self.config.block_size
        if len(virtual_kv_cache.block_table) < n_blocks:
            virtual_kv_cache.block_table += self.block_allocator.allocate(n_blocks - len(virtual_kv_cache.block_table))
        if len(virtual_kv_cache.block_table) < n_blocks:
            raise Exception('not enough kv cache')
        # 2. set vitual kv cache
        virtual_kv_cache.n_kv_cache_tokens = max(virtual_kv_cache.n_kv_cache_tokens, n_tokens)

        # 3. get phyical kv cache slot id
        return self.v2p(virtual_kv_cache, virtual_cache_ids)


    def free_blocks(self, virtual_kv_cache: VirtualKVCache, virtual_block_ids: list[int]):
        for virtual_block_id in sorted(virtual_block_ids, reverse=True):
            physical_block_id = virtual_kv_cache.block_table[virtual_block_id]
            self.block_allocator.free([physical_block_id])
            if virtual_block_id == len(virtual_kv_cache.block_tables) - 1:
                virtual_kv_cache.n_kv_cache_tokens -= (virtual_kv_cache.n_kv_cache_tokens + self.config.block_size - 1) % self.config.block_size + 1
            else:
                virtual_kv_cache.n_kv_cache_tokens -= self.config.block_size
            del virtual_kv_cache.block_table[virtual_block_id]


    def realloc(self, virtual_kv_cache: VirtualKVCache, n_tokens: int):
        if n_tokens > virtual_kv_cache.n_kv_cache_tokens:
            n_need_blocks = (n_tokens + self.config.block_size - 1) // self.config.block_size
            virtual_kv_cache.block_table += self.block_allocator.allocate(n_need_blocks - len(self.block_table))
            virtual_kv_cache.n_kv_cache_tokens = n_tokens
        else:
            n_need_blocks = (n_tokens + self.config.block_size - 1) // self.config.block_size
            self.block_allocator.free(virtual_kv_cache.block_table[n_need_blocks:])
            virtual_kv_cache.block_table = virtual_kv_cache.block_table[:n_need_blocks]
            virtual_kv_cache.n_kv_cache_tokens = n_tokens

    
    def mov(self, src_virtual_kv_cache: VirtualKVCache, src_virtual_cache_ids: list[int], dst_virtual_kv_cache: VirtualKVCache, dst_virtual_cache_ids: list[int]):
        src_slot_ids = self.v2p(src_virtual_kv_cache, src_virtual_cache_ids)
        dst_slot_ids = self.v2p(dst_virtual_kv_cache, dst_virtual_cache_ids)
        assert src_slot_ids.numel() == dst_slot_ids.numel(), f'{src_slot_ids.numel()} {dst_slot_ids.numel()}'
        assert src_slot_ids.dim() == 1, f'{src_slot_ids.dim()}'
        assert src_slot_ids.dtype == torch.int, f'{src_slot_ids.dtype}'
        assert dst_slot_ids.dim() == 1, f'{src_slot_ids.dim()}'
        assert dst_slot_ids.dtype == torch.int, f'{dst_slot_ids.dtype}'
        key_cache   = self.kv_cache_pool.view(self.num_blocks * self.block_size, self.num_kv_heads * self.head_size)
        value_cache = self.kv_cache_pool.value_cache.view(self.num_blocks * self.block_size, self.num_kv_heads * self.head_size)
        key_cache[dst_slot_ids, :] = key_cache[src_slot_ids, :]
        value_cache[dst_slot_ids, :] = value_cache[src_slot_ids, :]


    def get_kv_cache(self, virtual_kv_cache: VirtualKVCache) -> KVCache:
        return self.vid2kv_cache[virtual_kv_cache.vid]