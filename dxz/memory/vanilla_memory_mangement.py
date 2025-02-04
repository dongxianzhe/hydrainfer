import torch
from torch import Tensor
from dxz.memory.block_allocator import BlockAllocator
from dxz.memory.kv_cache import KVCache
from dxz.memory.virtual_kv_cache import VirtualKVCache
from dxz.memory.memory_management import MemoryConfig, MemoryContext, MemoryManagementUnit
from dxz.utils.allocate import IncreaingAllocator

try:
    from dxz._C.data_transfer import block_migration 
except ImportError:
    print('block migration lib import failed')


class VinallaMemoryManagementUnit(MemoryManagementUnit):
    def __init__(self, config: MemoryConfig, context: MemoryContext):
        self.config = config
        self.context = context
        assert self.config.memory_management_policy == 'vanilla'
        n_blocks_per_layer = config.num_blocks // context.n_layers
        # self.kv_cache = torch.randn(size=(context.n_layers, 2, n_blocks_per_layer, config.block_size, context.num_kv_heads, context.head_size), dtype=context.dtype, device=context.device)
        self.tensor_kv_caches: list[Tensor] = []
        self.kv_caches: list[KVCache] = []
        self.kv_caches_memory_handle: list[block_migration.cudaMemoryIpcHandle] = []
        for layer_id in range(context.n_layers):
            kv_cache = torch.randn(size=(2, n_blocks_per_layer, config.block_size, context.num_kv_heads, context.head_size), dtype=context.dtype, device=context.device)
            self.tensor_kv_caches.append(kv_cache)
            self.kv_caches.append(KVCache(key_cache=kv_cache[0, :, :, :, :], value_cache=kv_cache[1, :, :, :, :]))
            self.kv_caches_memory_handle.append(block_migration.get_ipc_mem_handle(kv_cache))

        self.block_allocator = BlockAllocator(n_blocks_per_layer)

        self.vid_allocator = IncreaingAllocator(first_value=1)

        for handle in self.kv_caches_memory_handle:
            print(f'handle {handle}')

    def allocate_virtual_kv_caches(self, n_virtual_kv_caches: int) -> list[VirtualKVCache]:
        assert n_virtual_kv_caches == self.context.n_layers, f"not supported other allocation yet {n_virtual_kv_caches}"
        virtual_kv_caches: list["VirtualKVCache"] = []
        for layer_id in range(n_virtual_kv_caches):
            virtual_kv_cache = VirtualKVCache(
                vid = self.vid_allocator.allocate(), 
                n_kv_cache_tokens = 0, 
                block_table = [], 
                layer_id = layer_id, 
                memory_handle=self.kv_caches_memory_handle[layer_id]
            )
            virtual_kv_caches.append(virtual_kv_cache)
        return virtual_kv_caches

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
        slot_ids: list[int] = []
        for vcid in virtual_cache_ids:
            block_id = vcid // self.config.block_size
            block_offset = vcid % self.config.block_size
            slot_id = virtual_kv_cache.block_table[block_id] * self.config.block_size + block_offset
            slot_ids.append(slot_id)
        return slot_ids

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
            virtual_kv_cache.block_table += self.block_allocator.allocate(n_need_blocks - len(virtual_kv_cache.block_table))
            virtual_kv_cache.n_kv_cache_tokens = n_tokens
        else:
            n_need_blocks = (n_tokens + self.config.block_size - 1) // self.config.block_size
            self.block_allocator.free(virtual_kv_cache.block_table[n_need_blocks:])
            virtual_kv_cache.block_table = virtual_kv_cache.block_table[:n_need_blocks]
            virtual_kv_cache.n_kv_cache_tokens = n_tokens
    
    def mov(self, src_virtual_kv_cache: VirtualKVCache, src_virtual_cache_ids: list[int], dst_virtual_kv_cache: VirtualKVCache, dst_virtual_cache_ids: list[int]):
        raise NotImplementedError

    def get_kv_cache(self, virtual_kv_cache: VirtualKVCache) -> KVCache:
        return self.kv_caches[virtual_kv_cache.layer_id]

    def migrate_blocks(self, src_virtual_kv_cache: VirtualKVCache, dst_virtual_kvcache: VirtualKVCache):
        # if src_memory_handle not in self.src_memory_handle_dict:
        #     block_migration.register_ipc_mem_handle(src_memory_handle)
        #     src_memory_handle_dict[]
        assert src_virtual_kv_cache.memory_handle is not None
        dev_ptr = block_migration.register_ipc_mem_handle(src_virtual_kv_cache.memory_handle)
        print(f'dev_ptr {dev_ptr}')
        block_migration.migrate_blocks(
            0, # prefill_start_head: int
            self.context.num_kv_heads, # prefill_end_head: int
            src_virtual_kv_cache.block_table, # prefill_block_indexes: list[int]
            0, # decoding_start_head: int
            self.context.num_kv_heads, # decoding_end_head:int
            dst_virtual_kvcache.block_table, # decoding_block_indexes: list[int]
            dev_ptr, # prefill_dev_ptr_index: int
            self.context.num_kv_heads, # num_heads: int
            self.tensor_kv_caches[dst_virtual_kvcache.layer_id], # decoding_worker_kv_cache: Tensor
        )
        torch.cuda.synchronize()
        print('torch.cuda.synchronize()')