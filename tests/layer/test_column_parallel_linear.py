import ray
import torch
from torch import Tensor, nn
from dataclasses import dataclass
from dxz.utils.ray_utils import launch_ray_cluster, get_ip_address
from dxz.model_parallel.process_group import ProcessGroup, init_global_process_group
from dxz.layer.linear import ColumnParallelLinear

@dataclass
class Config:
    hidden_size: int = 4096
    intermediate_size: int = 10084
    

@ray.remote(num_cpus=0, num_gpus=1)
class ColumnParallelLinearWorker:
    def __init__(self, rank: int, world_size: int, config: Config):
        self.config = config

        print(f'worker rank {rank} world_size {world_size}')
        init_global_process_group(backend='nccl', rank=rank, world_size=world_size, init_method='tcp://localhost:12359')
        self.process_group = ProcessGroup(rank=rank, global_ranks=[0, 1])

        dtype: torch.dtype = torch.half
        device: torch.device = torch.device('cuda:0')
        torch.set_default_device(device)
        torch.set_default_dtype(dtype)
        self.model = ColumnParallelLinear(in_features=config.hidden_size, out_features=config.hidden_size, bias=True, gather_output=True, process_group=self.process_group)
        print(f'worker {rank} init finished')

    def forward(self, x: Tensor) -> Tensor:
        o = self.model(x)
        return o

    def load_state_dict(self, state_dict: dict[str, Tensor]):
        self.model.load_state_dict(state_dict)


class ColumnParallelLinearEngine:
    def __init__(self, config: Config):
        launch_ray_cluster(is_head_node=True, head_node_ip=get_ip_address(), ray_cluster_port=8766, namespace='dxz')
        world_size = 2
        self.workers = [ColumnParallelLinearWorker.remote(rank=rank, world_size=world_size, config=config) for rank in range(world_size)]

    def forward(self, x: Tensor):
        objs = []
        for worker in self.workers:
            obj = worker.forward.remote(x)
            objs.append(obj)
        results = ray.get(objs)
        return results[0]
    
    def load_state_dict(self, state_dict: dict[str, Tensor]):
        objs = []
        for worker in self.workers:
            obj = worker.load_state_dict.remote(state_dict)
            objs.append(obj)
        ray.get(objs)


if __name__ == '__main__':
    # 1. settings
    n_tokens = 16
    config = Config()
    dtype: torch.dtype = torch.half
    device: torch.device = torch.device('cuda:0')
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)

    # 2. models
    model_ref = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=True)
    model_ref.to(dtype)
    model_ref.to(device)
    model_ref.eval()
    engine = ColumnParallelLinearEngine(config)

    # 3. load weight
    state_dict = {
        "weight" : torch.randn(size=(config.hidden_size, config.hidden_size), dtype=dtype, device=device), 
        "bias"   : torch.randn(size=(config.hidden_size, ), dtype=dtype, device=device), 
    }

    engine.load_state_dict(state_dict)
    model_ref.load_state_dict(state_dict)
    
    # 4. forward
    x = torch.randn((n_tokens, config.hidden_size), dtype=dtype, device=device)
    o = engine.forward(x)
    o_ref = model_ref(x)
    print(torch.allclose(o, o_ref, atol=1e-1, rtol=1e-2))
    print(f'o.view(-1)[:8]: {o.view(-1)[:8]}')
    print(f'o_ref.view(-1)[:8]: {o_ref.view(-1)[:8]}')