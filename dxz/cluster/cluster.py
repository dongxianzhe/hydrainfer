import ray
import torch
import asyncio
from dataclasses import dataclass, field, fields
from dxz.utils.ray_utils import launch_ray_cluster, get_ip_address
from dxz.engine.engine import EngineConfig
from dxz.utils.zmq_utils import init_zmq_recv, init_zmq_send
from dxz.request.request import Request
from dxz.cluster.rayepdnode import RayEPDNode, RayEPDNodeConfig, RayEPDNodeContext
from dxz.cluster.node_registry import NodeRegistry
from dxz.cluster.scheduler import RequestScheduler, RequestScheudlerConfig


@dataclass
class ClusterConfig:
    zmq_url: str
    num_nodes: int
    rayepdnode_config: RayEPDNodeConfig = field(default_factory=RayEPDNodeConfig)


class Cluster:
    def __init__(self, config: ClusterConfig):
        self.config = config
        # 1. node registry
        self.e_registry = NodeRegistry()
        # 2. scheduler
        self.e_scheduler = RequestScheduler(config=RequestScheudlerConfig())
        # 3. scale up nodes
        self.e_registry.register_scheduler(self.e_scheduler)

        for i in range(self.config.num_nodes):
            node = ray.remote(
                num_cpus=0,
                num_gpus=0.1,
                max_restarts=1,
                name=f'epdnode{i}',
                namespace='dxz',
                lifetime='detached'
            )(RayEPDNode).remote(
                config=self.config.rayepdnode_config, 
                context=RayEPDNodeContext(zmq_url=config.zmq_url)
            )
            node.step_loop.remote()
            self.e_registry.register_node(node)

    def add_request(self, request: Request):
        self.e_scheduler.add_request(request)


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
    num_nodes = 5, 
    rayepdnode_config = RayEPDNodeConfig(
        engine_config = EngineConfig(
            model_name='gpt2',  
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