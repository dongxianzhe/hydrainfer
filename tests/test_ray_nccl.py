import time
import ray
from ray.actor import ActorHandle
import torch
import torch.distributed as dist
from torch.distributed import P2POp, batch_isend_irecv
import socket

ray.init()
resources = ray.cluster_resources()
print(f'resources {resources} num_gpus {resources['GPU']}')

class GPUActor:
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        if self.rank == 0:
            self.buf = torch.arange(4, dtype=torch.int, device=torch.device('cuda:0'))
        else:
            self.buf = torch.zeros((4, ), dtype=torch.int, device=torch.device('cuda:0'))
    
    def get_init_method(self) -> str:
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        port = 45678
        nccl_init_method = f"tcp://{ip_address}:{port}"
        return nccl_init_method

    def init(self, init_method: str):
        dist.init_process_group(
            backend="nccl", 
            rank=self.rank, 
            world_size=self.world_size, 
            init_method=init_method, 
        )
        print(f'rank {self.rank} start warm up p2p operation')
        p2p_op_list = []
        if self.rank == 0:
            for i in range(1, self.world_size):
                tensor = torch.ones(size=(1, ), dtype=torch.int, device=torch.device('cuda:0'))
                p2p_op_list.append(P2POp(dist.isend, tensor, i))
        else:
            tensor = torch.zeros(size=(1, ), dtype=torch.int, device=torch.device('cuda:0'))
            p2p_op_list.append(P2POp(dist.irecv, tensor, 0))
        reqs = batch_isend_irecv(p2p_op_list)
        for req in reqs:
            req.wait()

        print(f'rank {self.rank} finished warm up p2p operation')


    def send_recv(self):
        print(f'rank {self.rank} before send {self.buf}')
        is_send = self.rank == 0
        op = dist.isend if is_send else dist.irecv
        rank = 1 if is_send else 0

        p2p_op_list: list[P2POp] = []
        p2p_op_list.append(P2POp(op, self.buf, rank))

        reqs = batch_isend_irecv(p2p_op_list)
        for req in reqs:
            req.wait()
        print(f'rank {self.rank} after send {self.buf}')


actor_1: ActorHandle[GPUActor] = ray.remote(num_gpus=1)(GPUActor).remote(rank=0, world_size=2)
actor_2: ActorHandle[GPUActor] = ray.remote(num_gpus=1)(GPUActor).remote(rank=1, world_size=2)

init_method = ray.get(actor_1.get_init_method.remote())
print(f'init_method {init_method}')

obj1 = actor_1.init.remote(init_method=init_method)
obj2 = actor_2.init.remote(init_method=init_method)
ray.get(obj1)
ray.get(obj2)

obj1 = actor_1.send_recv.remote()
obj2 = actor_2.send_recv.remote()
ray.get(obj1)
ray.get(obj2)

print('test finished')