import torch
from torch import Tensor
import torch.distributed as dist
from torch.distributed import P2POp, batch_isend_irecv
from dataclasses import dataclass, field
from typing import Optional, Literal
from dxz._C.data_transfer.block_migration import get_ipc_mem_handle
from dxz._C.data_transfer import block_migration
try:
    from dxz._C.kernel.cache_kernels import set_image_cache
except ImportError:
    set_image_cache = None
    print('set_image_cache import failed')
from dxz.memory import BlockAllocator
from dxz.utils.allocate import IncreaingAllocator
from dxz.utils.torch_utils import str2dtype, str2device


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
        self.block_size = caches[0].shape[1]
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

        if self.device.type == 'cuda' and set_image_cache:
            for cache, value in zip(self.caches, values):
                set_image_cache(
                    slot_ids, 
                    value,  
                    cache,
                )
            return

        for cache, value in zip(self.caches, values):
            slot_view = cache.view(-1, cache.shape[-2], cache.shape[-1])
            print(slot_view.shape, slot_ids.shape, value.shape)
            slot_view[slot_ids, :, :] = value
            return


@dataclass
class VirtualTokenCache:
    vid: int
    n_cache_tokens: int = 0
    block_table: list[int] = field(default_factory=list)
    memory_handle: Optional[list[int]] = None # used in cuda ipc memory handle
    rank: int = -1 # used in nccl


@dataclass
class TokenCacheBlockManagerConfig:
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

    def allocate_virtual_cache(self) -> VirtualTokenCache:
        return VirtualTokenCache(
            vid = self.vid_allocator.allocate(), 
            n_cache_tokens = 0, 
            block_table = [], 
            memory_handle = self.memory_handle,
            rank=self.rank,
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
            raise Exception(f'not enough cache, total n_blocks {self.n_blocks}, virtual cache need {n_blocks} allready has {len(virtual_cache.block_table)}')
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

    def migrate_blocks(self, src_virtual_cache: VirtualTokenCache, dst_virtual_cache: VirtualTokenCache, is_send: bool=False, backend: Literal["nccl", "ipc"]="ipc"):
        assert src_virtual_cache.n_cache_tokens == dst_virtual_cache.n_cache_tokens, f'{src_virtual_cache.n_cache_tokens} {dst_virtual_cache.n_cache_tokens}'
        if backend == 'ipc':
            assert is_send == False
            assert src_virtual_cache.memory_handle is not None
            with torch.cuda.stream(self.migrate_stream):
                block_migration.migrate_blocks(
                    src_virtual_cache.block_table, 
                    dst_virtual_cache.block_table, 
                    src_virtual_cache.memory_handle, 
                    self.cache_tensor,
                )
        elif backend == 'nccl':
            block_table = src_virtual_cache.block_table if is_send else dst_virtual_cache.block_table
            op = dist.isend if is_send else dist.irecv
            rank = dst_virtual_cache.rank if is_send else src_virtual_cache.rank

            p2p_op_list: list[P2POp] = []
            for block_id in block_table:
                for block_tensor in self.migrate_block_tensors_view[block_id]:
                    p2p_op_list.append(P2POp(op, block_tensor, rank))

            torch.cuda.set_device(torch.device('cuda:0'))
            with torch.cuda.stream(self.migrate_stream):
                reqs = batch_isend_irecv(p2p_op_list)
                for idx, req in enumerate(reqs):
                    req.wait()
        else:
            raise Exception(f'invalid block migrate backend {backend}')

    def synchronize(self):
        self.migrate_stream.synchronize()