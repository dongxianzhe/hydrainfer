import torch
import argparse
from typing import Literal
from dataclasses import dataclass, fields, field
from dxz.memory.kv_cache import KVCache
from dxz.memory.block_allocator import BlockAllocator
from dxz.memory.virtual_kv_cache import VirtualKVCache


@dataclass
class MemoryConfig:
    memory_management_policy: Literal['vanilla', 'shared'] = 'vanilla'
    num_blocks: int = 20000
    block_size: int = 16

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> 'MemoryConfig':
        attrs = [attr.name for attr in fields(cls)]
        config = cls(**{attr: getattr(args, attr) for attr in attrs})
        return config

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--memory-management-policy', type=str, choices=['vanilla', 'shared'], default='vanilla', help='Memory management policy.')
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
    def allocate_virtual_kv_caches(self, n_virtual_kv_caches: int) -> list[VirtualKVCache]:
        """
            allocate a set of virtual_kv_caches
            typically n_virtual_kv_caches is equal to model layers
            but in some token prunning cases, we may need additional temporary virtual_kv_cache to implementd kvcache mov, merge etc. operation
        """
        raise NotImplementedError

    def set(self, virtual_kv_cache: VirtualKVCache, virtual_cache_ids: list[int]) -> list[int]:
        """
            fill virtual kv cache at cache_ids with tokens logically), this operation allocate memory automatic if necessary
            return physical slot ids

            eg. suppose block size is 4
            a virtual kv cache may looks like as below
            n_kv_cache_tokens: 5
            virtual cache ids: [0, 1, 2, 3, 4]
            block table      : [3, 7]
            slot ids         : [0, 1, 2, 3, 28]

            and we want set virtual cache [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
            suppose the memory mangement allocate block 5 3 to this virtual kv cache
            after this function the virtual kv cache will be 
            n_kv_cache_tokens: 15
            virtual cache ids: [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14]
            block table      : [ 3,  7,  5,  3]
            slot ids         : [ 0,  1,  2,  3, 28, 29, 30, 31, 20, 21, 22, 23, 12, 13, 14]
        """
        raise NotImplementedError

    def free_blocks(self, virtual_kv_cache: VirtualKVCache, virtual_block_ids: list[int]):
        """
            free memory blocks allocated to virtual_kv_cache at virtual_block_ids

            eg.
            suppose we have a virtual kv cache with 15 tokens and we want free 1 block because of sliding window attention
            n_kv_cache_tokens: 15
            virtual cache ids: [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14]
            virtual block ids: [ 0,  1,  2,  3]
            block table      : [ 3,  7,  5,  3]
            slot ids         : [ 0,  1,  2,  3, 28, 29, 30, 31, 20, 21, 22, 23, 12, 13, 14]

            after free, the virtual kv cache will be
            n_kv_cache_tokens: 11
            virtual cache ids: [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
            virtual block ids: [ 0,  1,  2]
            block table      : [ 3,  5,  3]
            slot ids         : [ 0,  1,  2,  3, 20, 21, 22, 23, 12, 13, 14]
        """
        raise NotImplementedError

    def realloc(self, virtual_kv_cache: VirtualKVCache, n_tokens: int):
        """
            allocate or free memory for virtual_kv_cache to n_tokens
        """
        raise NotImplementedError

    def mov(self, src_virtual_kv_cache: VirtualKVCache, src_virtual_cache_ids: list[int], dst_virtual_kv_cache: VirtualKVCache, dst_virtual_cache_ids: list[int]):
        """
            copy data from src_virtual_kv_cache at src_virtual_cache_ids to dst_virtual_kv_cache at dst_virtual_cache_ids
        """
        raise NotImplementedError

    def get_kv_cache(self, virtual_kv_cache: VirtualKVCache) -> KVCache:
        raise NotImplementedError


def getMemoryManagementUnit(config: MemoryConfig, context: MemoryContext):
    if config.memory_management_policy == 'vanilla':
        from dxz.memory.vanilla_memory_mangement import VinallaMemoryManagementUnit
        return VinallaMemoryManagementUnit(config, context)
    else:
        from dxz.memory.shared_memory_management import SharedMemoryManagementUnit
        return SharedMemoryManagementUnit(config, context)