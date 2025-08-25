from typing import Mapping
import torch.cuda.nvtx as nvtx
from torch import Tensor
import ray
import time
import torch
import random
from torch import nn
from dataclasses import dataclass
import torch.distributed as dist
from functools import partial

nsight_config = {
    "t": "cuda,cudnn,cublas,nvtx",
    "o": "'tensor_parallel_worker_%p'",
    "stop-on-exit": "true",
    "cuda-memory-usage": "true",
    "gpu-metrics-devices": "all", 
}

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


class RowParallelLinear(nn.Module):
    """
        Linear layer with row parallelism.
            The linear layer is defined as Y = XA + b. A is parallelized along
            its first dimension and X along its second dimension as:
                    -   -
                    | A_1 |
                    | .   |
                A = | .   |       X = [X_1, ..., X_p]
                    | .   |
                    | A_p |
                    -   -
    """
    def __init__(self, in_features: int, out_features: int, bias: bool, input_is_parallelized: bool, process_group: ProcessGroup):
        super().__init__()
        self.tp_size = process_group.world_size
        self.tp_rank = process_group.rank
        self.process_group = process_group

        assert in_features % self.tp_size == 0, f"in_features {in_features} is not divisible by tp_size {self.tp_size}"
        in_features_per_partition = in_features // self.tp_size
        self.weight = nn.Parameter(torch.empty(size=(out_features, in_features_per_partition)), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(size=(out_features, )), requires_grad=False)
        else:
            self.bias = None
        self.input_is_parallelized = input_is_parallelized

    def forward(self, input: Tensor) -> Tensor:
        # input (n_tokens, input_features=hidden_size)
        if not self.input_is_parallelized:
            input = self.process_group.scatter_to_tensor_parallel_region(input) # (n_tokens, in_features_per_partition=hidden_size / tp_size)
        output = torch.nn.functional.linear(input, self.weight, bias=None) # (n_tokens, output_features=hidden_size)
        if self.tp_size > 1:
            output = self.process_group.reduce_from_tensor_parallel_region(output) # (n_tokens, output_features=hidden_size)
        # N.B. need to apply bias after the reduce
        if self.bias is not None:
            output.add_(self.bias) # (n_tokens, hidden_size)
        return output

    def load_state_dict(self, state_dict: dict[str, Tensor]):
        weight = state_dict['weight']
        assert weight.dim() == 2, f'weight dim should be 2 but got {weight.shape}'
        assert weight.shape[1] % self.tp_size == 0, f'weight {weight.shape} second dim is not divisible by tp_size {self.tp_size}'
        chunks = weight.chunk(self.tp_size, dim=1)
        self.weight.copy_(chunks[self.tp_rank])

        if self.bias is not None:
            bias = state_dict['bias']
            assert bias.dim() == 1, f'bias dim should be 1 but got {bias.shape}'
            self.bias.copy_(bias)


class ColumnParallelLinear(nn.Module):
    """
        Linear layer with column parallelism.
        The linear layer is defined as Y = XA + b. A is parallelized along
        its second dimension as A = [A_1, ..., A_p].
    """
    def __init__(self, in_features: int, out_features: int, bias: bool, gather_output: bool, process_group: ProcessGroup):
        super().__init__()
        self.tp_rank = process_group.rank
        self.tp_size = process_group.world_size
        self.process_group = process_group
        assert out_features % self.tp_size == 0, f"out_features {out_features} is not divisible by tp_size {self.tp_size}"
        out_features_per_partition = out_features // self.tp_size
        self.weight = nn.Parameter(torch.empty(size=(out_features_per_partition, in_features)), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(size=(out_features_per_partition, )), requires_grad=False)
        else:
            self.bias = None
        self.gather_output = gather_output
        
    def forward(self, input: Tensor) -> Tensor:
        output = torch.nn.functional.linear(input, self.weight, bias=self.bias)
        if self.tp_size > 1 and self.gather_output:
            output = self.process_group.gather_from_tensor_parallel_region(output)
        return output

    def load_state_dict(self, state_dict: dict[str, Tensor]):
        weight = state_dict['weight']
        assert weight.dim() == 2, f'weight dim should be 2 but got {weight.shape}'
        assert weight.shape[0] % self.tp_size == 0, f'weight {weight.shape} first dim is not divisible by tp_size {self.tp_size}'
        chunks = weight.chunk(self.tp_size, dim=0)
        self.weight.copy_(chunks[self.tp_rank])

        if self.bias is not None:
            bias = state_dict['bias']
            assert bias.dim() == 1, f'bias dim should be 1 but got {bias.shape}'
            assert bias.shape[0] % self.tp_size == 0, f'bias {bias.shape} is not disibile by tp_size {self.tp_size}'
            chunks = bias.chunk(self.tp_size, dim=0)
            self.bias.copy_(chunks[self.tp_rank])


@torch.inference_mode
def benchmark(forward_fn, warmup=3, n_iter=100):
    for _ in range(warmup):
        forward_fn()
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(n_iter):
        forward_fn()
        torch.cuda.synchronize()
    end = time.time()

    latency = (end - start) / n_iter
    return latency


@dataclass
class FakeModelConfig:
    vocab_size: int = 1024
    hidden_size = 4096
    intermediate_size = 8192


@ray.remote(num_cpus=1, num_gpus=1, runtime_env={"nsight": nsight_config})
class FakeModelWorker(nn.Module):
    def __init__(self, config: FakeModelConfig):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.silu = nn.SiLU()
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.head = nn.Linear(config.hidden_size, config.vocab_size)
        self.to(torch.half)
        self.to(torch.device('cuda:0'))
        self.eval()

    @torch.inference_mode()
    def forward(self, input_ids: list[int]) -> list[int]:
        x = torch.tensor(input_ids, dtype=torch.int, device=torch.device('cuda:0'))
        with nvtx.range("embed"):
            h = self.embed(x)
        for _ in range(32):
            with nvtx.range("up"):
                h = self.up(h)
            with nvtx.range("silu"):
                h = self.silu(h)
            with nvtx.range("down"):
                h = self.down(h)
        with nvtx.range("head"):
            logits = self.head(h)
        tokne_ids = torch.argmax(logits, dim=-1, keepdim=False)
        return tokne_ids.tolist()


class FakeModel(nn.Module):
    def __init__(self, config: FakeModelConfig):
        super().__init__()
        self.worker = FakeModelWorker.remote(config)
    
    @torch.inference_mode()
    def forward(self, input_ids: list[int]) -> list[int]:
        obj = self.worker.forward.remote(input_ids)
        o = ray.get(obj)
        return o


@ray.remote(num_cpus=1, num_gpus=1, runtime_env={"nsight": nsight_config})
class FakeModelTensorParallelWorker(nn.Module):
    def __init__(self, config: FakeModelConfig, rank: int, world_size: int):
        super().__init__()
        init_global_process_group(backend='nccl', world_size=world_size, rank=rank, init_method='tcp://localhost:12360')
        self.process_group = ProcessGroup(rank=rank, global_ranks=list(range(0, world_size)))
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.up = ColumnParallelLinear(in_features=config.hidden_size, out_features=config.intermediate_size, bias=True, gather_output=False, process_group=self.process_group)
        self.silu = nn.SiLU()
        self.down = RowParallelLinear(in_features=config.intermediate_size, out_features=config.hidden_size, bias=True, input_is_parallelized=True, process_group=self.process_group)
        self.head = nn.Linear(config.hidden_size, config.vocab_size)
        self.to(torch.half)
        self.to(torch.device('cuda:0'))
        self.eval()
        print(f'worker {rank} nccl init finished')

    @torch.inference_mode()
    def forward(self, input_ids: list[int]) -> list[int]:
        x = torch.tensor(input_ids, dtype=torch.int, device=torch.device('cuda:0'))
        with nvtx.range("embed"):
            h = self.embed(x)
        for _ in range(32):
            with nvtx.range("up column parallel"):
                h = self.up(h)
            with nvtx.range("silu"):
                h = self.silu(h)
            with nvtx.range("down row parallel"):
                h = self.down(h)
        with nvtx.range("head"):
            logits = self.head(h)
        tokne_ids = torch.argmax(logits, dim=-1, keepdim=False)
        return tokne_ids.tolist()


class FakeModelTensorParallel(nn.Module):
    def __init__(self, config: FakeModelConfig):
        super().__init__()
        self.config = config
        self.world_size = 2
        self.workers = [FakeModelTensorParallelWorker.remote(config, rank=i, world_size=self.world_size) for i in range(self.world_size)]

    def forward(self, input_ids: list[int]) -> list[int]:
        objs = []
        for worker in self.workers:
            obj = worker.forward.remote(input_ids)
            objs.append(obj)
        results = ray.get(objs)
        return results[0]


if __name__ == '__main__':
    config = FakeModelConfig()
    model = FakeModel(config)
    model_parallel = FakeModelTensorParallel(config) 

    n_tokens = 4096
    input_ids = [random.randint(0, config.vocab_size - 1) for _ in range(n_tokens)]

    model_forward = partial(model, input_ids=input_ids)
    model_parallel_forward = partial(model_parallel, input_ids=input_ids)

    print(f'model_forward latency {benchmark(model_forward)} s')
    print(f'model_parallel_forward latency {benchmark(model_parallel_forward)} s')