import asyncio
import torch
import ray
from dxz.utils.ray_utils import launch_ray_cluster, get_ip_address
from dxz.cluster.rayepdnode import RayEPDNode, RayEPDNodeConfig, RayEPDNodeContext
from dxz.engine.engine import EngineConfig
from dxz.utils.zmq_utils import init_zmq_recv, init_zmq_send
from dxz.request.request import Request
from dataclasses import dataclass, field, fields


class NodeRegistry:
    def __init__(self):
        pass
    def heartbeat_loop():
        pass
    def register_node():
        pass
    def unregister_node():
        pass
    def get_nodes():
        pass


@dataclass
class GlobalManagerConfig:
    pass


class GlobalManager:
    def __init__(self):
        pass


@dataclass
class ClusterConfig:
    zmq_url: str
    num_nodes: int
    rayepdnode_config: RayEPDNodeConfig = field(default_factory=RayEPDNodeConfig)


class Cluster:
    def __init__(self, config: ClusterConfig):
        self.config = config
        # 1. node actors creation
        self.nodes = []
        for i in range(self.config.num_nodes):
            node = ray.remote(
                num_cpus=0,
                num_gpus=0,
                max_restarts=1,
                name=f'epdnode{i}',
                namespace='dxz',
                lifetime='detached'
            )(RayEPDNode).remote(
                config=self.config.rayepdnode_config, 
                context=RayEPDNodeContext(zmq_url=config.zmq_url)
            )
            self.nodes.append(node)

        for node in self.nodes:
            node.step_loop.remote()
        # 2. node registry

        # 3. global manager
        self.next_node_id: int = 0

    def add_request(self, request: Request):
        print(f'send to node {self.next_node_id}')
        self.nodes[self.next_node_id].add_request.remote(request)
        self.next_node_id = (self.next_node_id + 1) % self.config.num_nodes


launch_ray_cluster(is_head_node=True, head_node_ip=get_ip_address(), ray_cluster_port=8765, namespace='dxz')
from dxz.request.request import Request, SamplingParameters
request = Request(
    request_id = 0, 
    prompt = f"what's the weather like today?", 
    image = None, 
    image_base64 = None, 
    sampling_params = SamplingParameters(max_tokens=10)
)

zmq_url = f"tcp://{get_ip_address()}:40832"
zmq_recv = init_zmq_recv(zmq_url=zmq_url)
cluster = Cluster(config=ClusterConfig(
    zmq_url = zmq_url, 
    num_nodes = 2, 
    rayepdnode_config = RayEPDNodeConfig(
        engine_config = EngineConfig(
            model_name='gpt2',  
            device=torch.device('cpu'), 
        )
    ), 
))

async def send_loop():
    while True:
        print('send request')
        cluster.add_request(request)
        await asyncio.sleep(3)


async def recv_loop():
    print('waiting loop start')
    while True:
        token_id = await zmq_recv.recv_pyobj()
        print(token_id)
        await asyncio.sleep(0.01)


async def main():
    loop1 = asyncio.create_task(send_loop())
    loop2 = asyncio.create_task(recv_loop())
    await asyncio.gather(loop1, loop2)

if __name__ == '__main__':
    asyncio.run(main())