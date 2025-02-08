from PIL import Image
import asyncio
import ray
from dataclasses import dataclass, field, fields
from dxz.utils.ray_utils import get_ip_address
from dxz.request import Request, OfflineInferenceOutput, SamplingParameters
from dxz.model import ModelFactoryConfig, getModelFactory, ModelFactoryContext
from dxz.engine import RequestProcessorConfig, BatchSchedulerConfig, ExecutorConfig, WorkerConfig, OfflineOutputTokenProcessor, RequestProcessParameters
from dxz.cluster import AsyncEPDNode
from dxz.utils.counter import Counter
from dxz.utils.zmq_utils import init_zmq_recv


@dataclass
class OfflineEPDDisaggregateEntryPointConfig:
    zmq_url: str = f"tcp://{get_ip_address()}:40832"
    request_processor_config: RequestProcessorConfig = field(default_factory=RequestProcessorConfig)
    model_factory_config: ModelFactoryConfig = field(default_factory=ModelFactoryConfig)
    batch_scheduler_config: BatchSchedulerConfig = field(default_factory=BatchSchedulerConfig)
    executor_config: ExecutorConfig = field(default_factory=ExecutorConfig)
    worker_config: WorkerConfig = field(default_factory=WorkerConfig)
    n_kv_blocks: int = 512
    n_image_blocks: int = 16


class OfflineEPDDisaggregateEntryPoint:
    def __init__(self, config: OfflineEPDDisaggregateEntryPointConfig):
        self.config = config
        self.zmq_recv = init_zmq_recv(zmq_url=config.zmq_url)
        self.enode = ray.remote(
            num_cpus=0,
            num_gpus=0.1,
            max_restarts=1,
            name=f'enode',
            namespace='dxz',
            lifetime='detached'
        )(AsyncEPDNode).remote(
            request_processor_config = config.request_processor_config, 
            model_factory_config = config.model_factory_config, 
            batch_scheduler_config = config.batch_scheduler_config, 
            executor_config = config.executor_config, 
            worker_config = config.worker_config, 
            n_kv_blocks = config.n_kv_blocks, 
            n_image_blocks = config.n_image_blocks, 
            enable_encode = True, 
            enable_prefill = False, 
            enable_decode = False, 
            zmq_send_url = None, 
        )
        self.pdnode = ray.remote(
            num_cpus=0,
            num_gpus=0.1,
            max_restarts=1,
            name=f'pdnode',
            namespace='dxz',
            lifetime='detached'
        )(AsyncEPDNode).remote(
            request_processor_config = config.request_processor_config, 
            model_factory_config = config.model_factory_config, 
            batch_scheduler_config = config.batch_scheduler_config, 
            executor_config = config.executor_config, 
            worker_config = config.worker_config, 
            n_kv_blocks = config.n_kv_blocks, 
            n_image_blocks = config.n_image_blocks, 
            enable_encode = False, 
            enable_prefill = True, 
            enable_decode = True, 
            zmq_send_url = None, 
        )
        self.nodes = [self.enode, self.pdnode]
        obj1 = self.enode.register_node.remote(self.pdnode)
        ray.get(obj1)
        for node in self.nodes:
            node.step_loop.remote()
            

    async def generate(self, requests: list[Request]) -> list[OfflineInferenceOutput]:
        finished = Counter()
        for request in requests:
            self.enode.add_request.remote(request, RequestProcessParameters(output_token_processors=[], print_output_text=True))
        outputs: list[OfflineInferenceOutput] = []
        while finished.value() < len(requests):
            output = await self.zmq_recv.recv_pyobj()
            outputs.append(output)
            finished.count()
        return outputs


async def main():
    requests = [
        # Request(
        #     request_id = 0, 
        #     prompt = f"<image>What is the content of this image?", 
        #     image = Image.open('./benchmark/dataset/cherry_blossom.jpg'), 
        #     image_base64 = None, 
        #     sampling_params = SamplingParameters(max_tokens=10)
        # ), 
        Request(
            request_id = 1, 
            prompt = f"<image>What is the content of this image?", 
            image = Image.open('./benchmark/dataset/cherry_blossom.jpg'), 
            image_base64 = None, 
            sampling_params = SamplingParameters(max_tokens=12)
        ), 
    ]
    config = OfflineEPDDisaggregateEntryPointConfig()
    config.batch_scheduler_config.debug_mode = True
    entrypoint = OfflineEPDDisaggregateEntryPoint(config) 
    outputs = await entrypoint.generate(requests=requests)
    for i, output in enumerate(outputs):
        print(f'output{i}: {output.text}')

if __name__ == '__main__':
    asyncio.run(main())