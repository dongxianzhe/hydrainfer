import ray
from dataclasses import dataclass, field, fields
from dxz.utils.ray_utils import get_ip_address
from dxz.model.model_factory import ModelFactoryConfig
from dxz.engine.scheduler import SchedulerConfig
from dxz.memory.memory_management import MemoryConfig
from dxz.engine.executor import ExecutorConfig
from dxz.engine.worker import WorkerConfig
from dxz.entrypoint.mllm import GenerateOutput
from dxz.utils.zmq_utils import init_zmq_recv
from dxz.cluster.enode import ENode, ENodeConfig, ENodeContext
from dxz.cluster.pnode import PNode, PNodeConfig, PNodeContext
from dxz.cluster.dnode import DNode, DNodeConfig, DNodeContext
from dxz.request.request import Request, SamplingParameters

@dataclass
class OfflineClusterConfig:
    zmq_url: str = f"tcp://{get_ip_address()}:40832"
    model_factory_config: ModelFactoryConfig = field(default_factory=ModelFactoryConfig)
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    memory_management_config: MemoryConfig = field(default_factory=MemoryConfig)
    executor_config: ExecutorConfig = field(default_factory=ExecutorConfig)
    worker_config: WorkerConfig = field(default_factory=WorkerConfig)
    enode_config: ENodeConfig
    pnode_config: PNodeConfig
    dnode_config: DNodeConfig


class OfflineCluster:
    def __init__(self, config: OfflineClusterConfig):
        self.config = config
        self.zmq_recv = init_zmq_recv(zmq_url=config.zmq_url)
        self.enode = ray.remote(
            num_cpus=0,
            num_gpus=0.1,
            max_restarts=1,
            name=f'enode',
            namespace='dxz',
            lifetime='detached'
        )(ENode).remote(config.enode_config, ENodeContext(
            
        ))
        self.pnode = ray.remote(
            num_cpus=0,
            num_gpus=0.1,
            max_restarts=1,
            name=f'pnode',
            namespace='dxz',
            lifetime='detached'
        )(PNode).remote(config.pnode_config, PNodeContext(
            model_factory_config = config.model_factory_config, 
            memory_config = config.memory_management_config, 
            scheduler_config = config.scheduler_config,
            executor_config = config.executor_config, 
            worker_config = config.worker_config
        ))
        self.dnode = ray.remote(
            num_cpus=0,
            num_gpus=0.1,
            max_restarts=1,
            name=f'dnode',
            namespace='dxz',
            lifetime='detached'
        )(DNode).remote(config.dnode_config, DNodeContext(
            model_factory_config = config.model_factory_config, 
            memory_config = config.memory_management_config, 
            scheduler_config = config.scheduler_config,
            executor_config = config.executor_config, 
            worker_config = config.worker_config
        ))
        self.nodes = [self.enode, self.pnode, self.dnode]
        obj1 = self.enode.register_pnode.remote(self.pnode)
        obj2 = self.pnode.register_dnode.remote(self.dnode)
        ray.get([obj1, obj2])
        for node in self.nodes:
            node.step_loop.remote()

    async def generate(self, requests: list[Request]) -> list[GenerateOutput]:
        for request in requests:
            self.enode.add_request.remote(request)
        outputs: list[GenerateOutput] = []
        while self.finished.value() < len(requests):
            print('waiting loop')
            output = await self.zmq_recv.recv_pyobj()
            outputs.append(output)
            self.finished.count()
        return [output]

async def main():
    request = Request(
        request_id = 0, 
        prompt = f"What is the content of this image?", 
        image = None, 
        image_base64 = None, 
        sampling_params = SamplingParameters(max_tokens=10)
    )
    config = OfflineClusterConfig()
    config.model_factory_config.model_name='gpt2'
    config.scheduler_config.debug_mode = True
    mllm = OfflineCluster(config) 
    outputs = await mllm.generate(requests=[request])
    for output in outputs:
        print(output.text)

if __name__ == '__main__':
    asyncio.run(main())