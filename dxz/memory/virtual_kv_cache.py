import torch
from dataclasses import dataclass
from dxz.memory.kv_cache import KVCache
from dxz.memory.block_allocator import BlockAllocator

@dataclass
class MemoryConfig:
    num_blocks: int = 10000
    block_size: int = 16
     

@dataclass
class MemoryContext:
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    device: torch.device

class MemoryManagementUnit:
    """_summary_
        MemoryManagementUnit manage 
        1. cpu memory todo
        2. gpu memory
        3. vitual kv cache creation todo
    """
    def __init__(self, config: MemoryConfig, context: MemoryContext):
        self.config = config
        self.context = context
        self.kv_cache = KVCache(
            config.num_blocks, 
            config.block_size, 
            context.num_kv_heads, 
            context.head_size, 
            context.dtype, 
            context.device
        )
        self.kv_cache_allocator = BlockAllocator(config.num_blocks)

    def allocate(self, n_blocks: int) -> list[int]:
        return self.kv_cache_allocator.allocate(n_blocks)
    
    def free(self, blocks: list[int]) -> bool:
        return self.kv_cache_allocator.free(blocks)

    def allocate_virtual_kv_caches(self, n_virtual_kv_caches: int) -> list["VirtualKVCache"]:
        return [VirtualKVCache(self) for _ in range(n_virtual_kv_caches)]

    def v2p(self, cache_ids: list[int], block_table: list[int]) -> list[int]:
        slot_ids: list[int] = []
        for vcid in cache_ids:
            block_id = vcid // self.config.block_size
            block_offset = vcid % self.config.block_size
            slot_id = block_table[block_id] * self.config.block_size + block_offset
            slot_ids.append(slot_id)
        return slot_ids

    def move_physical_kv_caches(self, src_slot_ids: list[int], dst_slot_ids: list[int]):
        self.kv_cache.move_kv_cache(src_slot_ids, dst_slot_ids)

class VirtualKVCache:
    def __init__(self, mmu: MemoryManagementUnit):
        self.config: MemoryConfig = mmu.config
        self.block_size = self.config.block_size
        self.mmu = mmu

        self.n_kv_cache_tokens: int = 0
        self.block_tables: list[int] = []

    def set(self, virtual_cache_ids: list[int]) -> bool:
        """
            fill kv cache of cache_ids with tokens, cache_ids (n_tokens, )
            will allocate memory automatic if necessary
            return True means set success
            False means set failed due to not enough memory
        """
        # 1. try to allocate memory if block is not enough
        n_tokens = max(virtual_cache_ids) + 1
        n_blocks = (n_tokens + self.block_size - 1) // self.block_size
        if len(self.block_tables) < n_blocks:
            self.block_tables += self.mmu.allocate(n_blocks - len(self.block_tables))
        if len(self.block_tables) < n_blocks:
            raise Exception('not enough kv cache')
            return False
        # 2. set vitual kv cache
        self.n_kv_cache_tokens = max(self.n_kv_cache_tokens, n_tokens)
        return True

    def free_blocks(self, virtual_block_ids: list[int]) -> bool:
        for virtual_block_id in sorted(virtual_block_ids, reverse=True):
            physical_block_id = self.block_tables[virtual_block_id]
            self.mmu.free([physical_block_id])
            del self.block_tables[virtual_block_id]
        return True

    def realloc(self, n_tokens: int):
        if n_tokens > self.n_kv_cache_tokens:
            n_need_blocks = (n_tokens + self.block_size - 1) // self.block_size
            self.block_tables += self.mmu.allocate(n_need_blocks - len(self.block_tables))
            self.n_kv_cache_tokens = n_tokens
        else:
            n_need_blocks = (n_tokens + self.block_size - 1) // self.block_size
            self.mmu.free(self.block_tables[n_need_blocks:])
            self.block_tables = self.block_tables[:n_need_blocks]
            self.n_kv_cache_tokens = n_tokens