import torch
import argparse
from torch import Tensor
import torch.distributed as dist
from dataclasses import dataclass, field, fields


@dataclass
class ParallelConfig:
    """Configuration for the distributed execution.

    Args:
        tensor_parallel_size: number of tensor parallel groups.
        pipeline_parallel_size: number of pipeline parallel groups.
        tensor_parallel_rank: rank in the tensor parallel group.
        pipeline_parallel_rank: rank in the pipeline parallel group.
    """
    tp_size: int = 1
    pp_size: int = 1
    # tp_rank: int = 0
    # pp_rank: int = 0

    @property
    def world_size(self) -> int:
        return self.pp_size * self.tp_size

    @property
    def is_parallel(self) -> bool:
        return self.world_size > 1

    @property
    def is_last_stage(self) -> bool:
        return self.pp_rank == (self.pp_size - 1)



def init_global_process_group(backend='nccl', world_size=1, rank=0, init_method='env://'):
    """
    if use 'env://' as init method, the following environment variables should be set:
        RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
    alse can use 'tcp://<master_ip>:<port>' as init method
    """
    dist.init_process_group(
        backend=backend,   # ('nccl'，'gloo'，'mpi')
        world_size=world_size,  # num process
        rank=rank,  # process rank
        init_method=init_method
    )


class ProcessGroup:
    """
    create process group from global ranks
    eg.
        if we have 8 GPUs then global ranks are [0, 1, 2, 3, 4, 5, 6, 7]
        if we want to divide 8 process to 2 group, each group has 4 GPUs, then we can use:
        process0 calls ProcessGroup(rank=0, global_ranks=[0, 1, 2, 3])
        process1 calls ProcessGroup(rank=1, global_ranks=[0, 1, 2, 3])
        process2 calls ProcessGroup(rank=2, global_ranks=[0, 1, 2, 3])
        process3 calls ProcessGroup(rank=3, global_ranks=[0, 1, 2, 3])
        process4 calls ProcessGroup(rank=0, global_ranks=[4, 5, 6, 7])
        process5 calls ProcessGroup(rank=1, global_ranks=[4, 5, 6, 7])
        process6 calls ProcessGroup(rank=2, global_ranks=[4, 5, 6, 7])
        process7 calls ProcessGroup(rank=3, global_ranks=[4, 5, 6, 7])
    """
    def __init__(self, rank: int, global_ranks: list[int]):
        self.rank = rank
        self.world_size = len(global_ranks)
        # todo for now we only support tp and ignore pp in config
        self.process_group = dist.new_group(ranks=global_ranks)

    def all_reduce(self, tensor: Tensor, op=dist.ReduceOp.SUM):
        dist.all_reduce(input, op=op, group=self.process_group)

    def scatter_to_tensor_parallel_region(self, input: Tensor) -> Tensor:
        if self.world_size == 1:
            return input
        last_dim_size = input.shape[-1]
        assert last_dim_size % self.world_size == 0, f"last_dim_size {last_dim_size} is not divisible by world_size {self.world_size}"
        # torch.split does not create contiguous tensors by default
        t = input.split(split_size=last_dim_size // self.world_size, dim=-1) # a world_size shape tuple, each element is a tensor
        return t[self.rank]

    def reduce_from_tensor_parallel_region(self, input: Tensor) -> Tensor:
        if self.world_size == 1:
            return input
        dist.all_reduce(input, op=dist.ReduceOp.SUM, group=self.process_group)
        return input

    def gather_from_tensor_parallel_region(self, input: Tensor) -> Tensor:
        if self.world_size == 1:
            return input
        tensors = [torch.empty_like(input) for _ in range(self.world_size)]
        dist.all_gather(tensors, input, group=self.process_group)
        return torch.cat(tensors, dim=-1).contiguous()

class ParallelContext:
    tp_rank: int
    pp_rank: int
    process_group: ProcessGroup