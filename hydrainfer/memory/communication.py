import torch
from torch import Tensor
from typing import Literal
from hydrainfer.memory.token_cache import VirtualTokenCache
from hydrainfer.utils.logger import getLogger
from dataclasses import dataclass
logger = getLogger(__name__)

try:
    from hydrainfer._C.data_transfer.block_migration import get_ipc_mem_handle
    from hydrainfer._C.data_transfer import block_migration
except Exception as e:
    logger.warning('ipc block migration kernel import failed')
    get_ipc_mem_handle = None
    block_migration = None


class CommunicationBackend:
    def migrate_blocks(self, src_virtual_cache: VirtualTokenCache, dst_virtual_cache: VirtualTokenCache, is_send: bool):
        raise NotImplementedError()


class IPCHandleMemoryBackend(CommunicationBackend):
    def __init__(self, migrate_stream: torch.cuda.Stream, cache: Tensor, n_blocks: int, debug: bool=False):
        super().__init__()
        self.migrate_stream = migrate_stream
        self.cache = cache
        self.n_blocks = n_blocks
        self.debug = debug

    def migrate_blocks(self, src_virtual_cache: VirtualTokenCache, dst_virtual_cache: VirtualTokenCache, is_send: bool):
        with torch.cuda.stream(self.migrate_stream):
            if is_send:
                pass
            else:
                if self.debug:
                    logger.info(f'migrate blocks from rank {src_virtual_cache.rank} to rank {dst_virtual_cache.rank} via ipc')
                assert src_virtual_cache.memory_handle is not None
                block_migration.migrate_blocks(
                    src_virtual_cache.block_table, 
                    dst_virtual_cache.block_table, 
                    src_virtual_cache.memory_handle, 
                    self.cache,
                    src_virtual_cache.n_blocks_of_cache_manager, 
                )


import torch.distributed as dist
from torch.distributed import P2POp, batch_isend_irecv
class NCCLBackend(CommunicationBackend):
    def __init__(self, migrate_stream: torch.cuda.Stream, migrate_block_tensors_view: list[list[Tensor]], debug: bool=False):
        super().__init__()
        self.migrate_stream = migrate_stream
        self.migrate_block_tensors_view = migrate_block_tensors_view # block_id -> list of tensor
        self.debug = debug

    def migrate_blocks(self, src_virtual_cache: VirtualTokenCache, dst_virtual_cache: VirtualTokenCache, is_send: bool):
        block_table = src_virtual_cache.block_table if is_send else dst_virtual_cache.block_table
        op = dist.isend if is_send else dist.irecv
        rank = dst_virtual_cache.rank if is_send else src_virtual_cache.rank

        if self.debug and not is_send:
            logger.info(f'migrate blocks from rank {src_virtual_cache.rank} to rank {dst_virtual_cache.rank} via nccl')

        p2p_op_list: list[P2POp] = []
        for block_id in block_table:
            for block_tensor in self.migrate_block_tensors_view[block_id]:
                p2p_op_list.append(P2POp(op, block_tensor, rank))

        torch.cuda.set_device(torch.device('cuda:0'))
        with torch.cuda.stream(self.migrate_stream):
            reqs = batch_isend_irecv(p2p_op_list)
            for idx, req in enumerate(reqs):
                req.wait()


@dataclass
class CommunicationBackendManagerContext:
    migrate_stream: torch.cuda.Stream
    cache: Tensor # (n_layers, n_tokens, n_blocks, block_size, n_heads, head_size)
    n_blocks: int
    migrate_block_tensors_view: list[list[Tensor]]
    rank2host: dict[int, str]


def get_migrate_backend(backend: str, context: CommunicationBackendManagerContext, debug: bool=False) -> CommunicationBackend:
    if backend == 'ipc':
        return IPCHandleMemoryBackend(migrate_stream=context.migrate_stream, cache=context.cache, n_blocks=context.n_blocks, debug=debug)
    elif backend == 'nccl':
        return NCCLBackend(migrate_stream=context.migrate_stream, migrate_block_tensors_view=context.migrate_block_tensors_view, debug=debug)
    else:
        raise Exception(f'invalidd migrate backend {backend}')
        

@dataclass
class CommunicationBackendManagerConfig:
    intranode_migrate_backend: Literal['auto', 'ipc', 'nccl'] = 'auto'
    internode_migrate_backend: Literal['nccl'] = 'nccl'
    debug: bool=False


class CommunicationBackendManager(CommunicationBackend):
    def __init__(self, config: CommunicationBackendManagerConfig, context: CommunicationBackendManagerContext):
        self.context = context
        self.rank2host = context.rank2host
        if config.intranode_migrate_backend == 'auto':
            config.intranode_migrate_backend = 'ipc' if block_migration is not None else 'nccl'
            logger.info(f'auto set intra node migrate backend {config.intranode_migrate_backend}')

        self.intranode_backend = get_migrate_backend(config.intranode_migrate_backend, context, config.debug)
        self.internode_backend = get_migrate_backend(config.internode_migrate_backend, context, config.debug)

    def in_same_machine(self, rank1: int, rank2: int) -> bool:
        if rank1 not in self.rank2host or rank2 not in self.rank2host:
            return False
        return self.rank2host.get(rank1) == self.rank2host.get(rank2)

    def migrate_blocks(self, src_virtual_cache: VirtualTokenCache, dst_virtual_cache: VirtualTokenCache, is_send: bool):
        assert src_virtual_cache.n_cache_tokens == dst_virtual_cache.n_cache_tokens, f'{src_virtual_cache.n_cache_tokens} {dst_virtual_cache.n_cache_tokens}'
        if self.in_same_machine(src_virtual_cache.rank, dst_virtual_cache.rank):
            self.intranode_backend.migrate_blocks(src_virtual_cache, dst_virtual_cache, is_send)
        else:
            self.internode_backend.migrate_blocks(src_virtual_cache, dst_virtual_cache, is_send)