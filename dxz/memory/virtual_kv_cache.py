import torch
from dataclasses import dataclass, fields
from dxz.memory.kv_cache import KVCache
from dxz.memory.block_allocator import BlockAllocator
from typing import Literal
import argparse

@dataclass
class MemoryConfig:
    memory_management_policy: Literal['vanilla', 'shared_layers'] = 'vanilla'
    num_blocks: int = 20000
    block_size: int = 16

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> 'MemoryConfig':
        attrs = [attr.name for attr in fields(cls)]
        config = cls(**{attr: getattr(args, attr) for attr in attrs})
        return config

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--memory-management-policy', type=str, choices=['vanilla', 'shared_layers'], default='vanilla', help='Memory management policy.')
        parser.add_argument('--num-blocks', type=int, default=20000, help='Number of GPU kvcache blocks.')
        parser.add_argument('--block-size', type=int, default=16, help='Size of each block.')
        return parser


@dataclass
class MemoryContext:
    n_layers : int
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
        if self.config.memory_management_policy == 'vanilla':
            self.kv_caches: list[KVCache] = [
                KVCache(
                    config.num_blocks // self.context.n_layers,
                    config.block_size, 
                    context.num_kv_heads, 
                    context.head_size, 
                    context.dtype, 
                    context.device
                ) for _ in range(self.context.n_layers)
            ]
        else:
            self.kv_caches: list[KVCache] = [
                KVCache(
                    config.num_blocks, 
                    config.block_size, 
                    context.num_kv_heads, 
                    context.head_size, 
                    context.dtype, 
                    context.device
                )
            ]


    def allocate_virtual_kv_caches(self, n_virtual_kv_caches: int) -> list["VirtualKVCache"]:
        if self.config.memory_management_policy == 'vanilla':
            assert n_virtual_kv_caches % self.context.n_layers == 0
            return [VirtualKVCache(self.config, self.kv_caches[layer_id]) for layer_id in range(n_virtual_kv_caches)]
        elif self.config.memory_management_policy == 'shared_layers':
            return [VirtualKVCache(self.config, self.kv_caches[0]) for layer_id in range(n_virtual_kv_caches)]

class VirtualKVCache:
    def __init__(self, config: MemoryConfig, kv_cache: KVCache):
        self.config: MemoryConfig = config
        self.block_size = self.config.block_size
        self.kv_cache = kv_cache

        self.n_kv_cache_tokens: int = 0
        self.block_table: list[int] = []

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
        if len(self.block_table) < n_blocks:
            self.block_table += self.kv_cache.allocate(n_blocks - len(self.block_table))
        if len(self.block_table) < n_blocks:
            raise Exception('not enough kv cache')
            return False
        # 2. set vitual kv cache
        self.n_kv_cache_tokens = max(self.n_kv_cache_tokens, n_tokens)
        return True

    def free_blocks(self, virtual_block_ids: list[int]) -> bool:
        for virtual_block_id in sorted(virtual_block_ids, reverse=True):
            physical_block_id = self.block_table[virtual_block_id]
            self.kv_cache.free([physical_block_id])
            del self.block_table[virtual_block_id]
        return True

    def realloc(self, n_tokens: int):
        if n_tokens > self.n_kv_cache_tokens:
            n_need_blocks = (n_tokens + self.block_size - 1) // self.block_size
            self.block_table += self.mmu.allocate(n_need_blocks - len(self.block_table))
            self.n_kv_cache_tokens = n_tokens
        else:
            n_need_blocks = (n_tokens + self.block_size - 1) // self.block_size
            self.kv_cache.free(self.block_table[n_need_blocks:])
            self.block_table = self.block_table[:n_need_blocks]
            self.n_kv_cache_tokens = n_tokens

    def v2p(self, virtual_cache_ids: list[int]) -> list[int]:
        slot_ids: list[int] = []
        for vcid in virtual_cache_ids:
            block_id = vcid // self.config.block_size
            block_offset = vcid % self.config.block_size
            slot_id = self.block_table[block_id] * self.config.block_size + block_offset
            slot_ids.append(slot_id)
        return slot_ids