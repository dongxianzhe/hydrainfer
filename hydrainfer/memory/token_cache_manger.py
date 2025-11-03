import torch
from torch import Tensor
import torch.distributed as dist
from torch.distributed import P2POp, batch_isend_irecv
from typing import Optional, Literal
from dataclasses import dataclass, field
from hydrainfer.memory import VirtualTokenCache, TokenCache
from hydrainfer.utils.logger import getLogger
from hydrainfer.memory import BlockAllocator, BlockAllocatorMetrics
from hydrainfer.utils.allocate import IncreaingAllocator
from hydrainfer.utils.torch_utils import str2dtype, str2device, get_dtype_size
from hydrainfer.memory import CommunicationBackendManager, CommunicationBackendManagerConfig, CommunicationBackendManagerContext
from hydrainfer.memory import SharedCache, SharedCacheConfig

logger = getLogger(__name__)

try:
    from hydrainfer._C.data_transfer.block_migration import get_ipc_mem_handle
    from hydrainfer._C.data_transfer import block_migration
except Exception as e:
    logger.warning('ipc block migration kernel import failed')
    get_ipc_mem_handle = None
    block_migration = None


@dataclass
class TokenCacheManagerMetrics:
    allocator_metrics: BlockAllocatorMetrics
    cache_hit_rate: float


@dataclass
class TokenCacheBlockManagerConfig:
    communication_backend_manager_config: CommunicationBackendManagerConfig
    n_layers: int = 32
    n_tokens: int = 2
    n_blocks: int = 1024
    block_size: int = 16
    n_heads: int = 32
    head_size: int = 128
    dtype: str = "fp16"
    device: str = "cuda:0"


@dataclass
class TokenCacheBlockManagerContext:
    rank: int
    rank2host: dict[int, str]


class TokenCacheBlockManager:
    def __init__(self, config: TokenCacheBlockManagerConfig, context: TokenCacheBlockManagerContext):
        self.config = config
        self.context = context
        self.n_layers   = config.n_layers
        self.n_tokens   = config.n_tokens 
        self.n_blocks   = config.n_blocks
        self.block_size = config.block_size
        self.n_heads    = config.n_heads
        self.head_size  = config.head_size
        self.dtype      = str2dtype(config.dtype)
        self.device     = str2device(config.device)
        self.rank = context.rank

        self.cache_tensor = torch.randn(size=(self.n_layers, self.n_tokens, self.n_blocks, self.block_size, self.n_heads, self.head_size), dtype=self.dtype, device=self.device)

        # we create a list of list of tensor to store the block tensors view for migration because tensor slice is very slow
        self.migrate_block_tensors_view: list[list[Tensor]] = [[] for block_id in range(self.n_blocks)]
        for block_id in range(self.n_blocks):
            for layer_id in range(self.n_layers):
                for token_id in range(self.n_tokens):
                    self.migrate_block_tensors_view[block_id].append(self.cache_tensor[layer_id, token_id, block_id, :, :, :])

        self.memory_handle: list[int] = get_ipc_mem_handle(self.cache_tensor)
        self.block_allocator = BlockAllocator(self.n_blocks)
        self.vid_allocator = IncreaingAllocator(first_value=1)

        self.migrate_stream:torch.cuda.Stream = torch.cuda.Stream()

        self.migrate_manager = CommunicationBackendManager(
            self.config.communication_backend_manager_config, 
            CommunicationBackendManagerContext(
                migrate_stream=self.migrate_stream, 
                cache=self.cache_tensor, 
                n_blocks=self.n_blocks, 
                migrate_block_tensors_view=self.migrate_block_tensors_view, 
                rank2host=context.rank2host, 
            ))
        self.shared_cache = SharedCache(config=SharedCacheConfig(n_blocks=self.n_blocks))

        self.total_block_queried = 0.
        self.total_block_matched = 0.

    def get_num_avaiable_blocks(self) -> int:
        return self.block_allocator.get_num_avaiable_blocks() + self.shared_cache.get_num_avaiable_blocks()

    def _allocate_new_blocks(self, n_blocks: int) -> list[int]:
        block_ids = self.block_allocator.allocate(n_blocks)
        if len(block_ids) < n_blocks:
            block_ids += self.shared_cache.allocate(n_blocks - len(block_ids))
        self.shared_cache.pin(block_ids)
        return block_ids
        

    def allocate_virtual_cache(self, hashes: Optional[list[int]]=None) -> VirtualTokenCache:
        if hashes is None:
            n_cached_tokens = 0
            matched_block_ids = [] 
        else:
            block_ids: list[int] = self.shared_cache.match(hashes)
            matched_block_ids: list[int] = block_ids[: block_ids.index(-1) if -1 in block_ids else len(block_ids)]
            self.shared_cache.pin(matched_block_ids)
            n_cached_tokens = len(matched_block_ids) * self.block_size
            self.total_block_matched += len(matched_block_ids)
            self.total_block_queried += len(hashes)
        return VirtualTokenCache(
            vid = self.vid_allocator.allocate(), 
            n_cache_tokens = n_cached_tokens, 
            block_table = matched_block_ids, 
            memory_handle = self.memory_handle,
            rank=self.rank,
            n_blocks_of_cache_manager=self.n_blocks
        )

    def v2p(self, virtual_cache: VirtualTokenCache, virtual_cache_ids: list[int]) -> list[int]:
        physical_cache_ids: list[int] = []
        for vcid in virtual_cache_ids:
            block_id = vcid // self.block_size
            block_offset = vcid % self.block_size
            slot_id = virtual_cache.block_table[block_id] * self.block_size + block_offset
            physical_cache_ids.append(slot_id)
        return physical_cache_ids

    def set_blocks(self, virtual_cache: VirtualTokenCache, virtual_block_ids: list[int], hashes: list[int]):
        assert len(virtual_block_ids) == len(hashes)
        phyical_block_ids: list[int] = [virtual_cache.block_table[virtual_block_id] for virtual_block_id in virtual_block_ids]
        self.shared_cache.insert(hashes=hashes, block_ids=phyical_block_ids)

    def set(self, virtual_cache: VirtualTokenCache, virtual_cache_ids: list[int], hashes: list[int]):
        assert len(virtual_cache_ids) == len(hashes)
        assert max(virtual_cache_ids) + 1 < virtual_cache.n_cache_tokens
        physical_block_ids: list[int] = []
        for virtual_cache_id in virtual_cache_ids:
            virtual_block_id = virtual_cache_id // self.block_size
            physical_block_id = virtual_cache.block_table[virtual_block_id]
            assert self.shared_cache.is_write_safe(physical_block_id), 'copy on write is not supported now'
        self.shared_cache.insert(hashes=hashes, block_ids=physical_block_ids)

    def realloc(self, virtual_cache: VirtualTokenCache, n_tokens: int) -> bool:
        if n_tokens > virtual_cache.n_cache_tokens:
            n_need_blocks = (n_tokens + self.block_size - 1) // self.block_size
            virtual_cache.block_table += self._allocate_new_blocks(n_need_blocks - len(virtual_cache.block_table))
            if len(virtual_cache.block_table) * self.block_size < n_tokens:
                virtual_cache.n_cache_tokens = len(virtual_cache.block_table) * self.block_size
                return False
            virtual_cache.n_cache_tokens = n_tokens
            return True
        else:
            n_need_blocks = (n_tokens + self.block_size - 1) // self.block_size
            self.shared_cache.unpin(virtual_cache.block_table[n_need_blocks:])
            virtual_cache.block_table = virtual_cache.block_table[:n_need_blocks]
            virtual_cache.n_cache_tokens = n_tokens
            return True

    def get_layer_cache(self, layer_id: int) -> TokenCache:
        return TokenCache([self.cache_tensor[layer_id, token_id, :, :, :, :] for token_id in range(self.n_tokens)])

    def migrate_blocks(self, src_virtual_cache: VirtualTokenCache, dst_virtual_cache: VirtualTokenCache, is_send: bool=False):
        self.migrate_manager.migrate_blocks(src_virtual_cache, dst_virtual_cache, is_send)

    def synchronize(self):
        self.migrate_stream.synchronize()

    @classmethod
    def compute_n_blocks(cls, config: TokenCacheBlockManagerConfig, memory: int):
        return memory // (config.n_layers * config.n_tokens * config.block_size * config.n_heads * config.head_size * get_dtype_size(str2dtype(config.dtype)))

    def get_metrics(self) -> TokenCacheManagerMetrics:
        return TokenCacheManagerMetrics(
            allocator_metrics=self.block_allocator.get_metrics(), 
            cache_hit_rate=self.total_block_matched / self.total_block_queried if self.total_block_queried 
            != 0 else 0., 
        )