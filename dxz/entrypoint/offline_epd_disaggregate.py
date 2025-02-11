import ray
import asyncio
import argparse
from PIL import Image
from dataclasses import dataclass, field, fields
from dxz.utils.ray_utils import get_ip_address
from dxz.request import Request, OfflineInferenceOutput, SamplingParameters
from dxz.model import ModelFactoryConfig, getModelFactory, ModelFactoryContext
from dxz.engine import RequestProcessorConfig, BatchSchedulerConfig, ExecutorConfig, WorkerConfig, OfflineOutputTokenProcessor, RequestProcessParameters
from dxz.cluster import AsyncEPDNode, NodeConfig
from dxz.utils.counter import Counter
from dxz.utils.zmq_utils import init_zmq_recv
from dxz.utils.config_util import CLIConfig


@dataclass
class OfflineEPDDisaggregateEntryPointConfig(CLIConfig):
    zmq_url: str = f"tcp://{get_ip_address()}:40832"
    enode_config: NodeConfig = field(default_factory=NodeConfig)
    pnode_config: NodeConfig = field(default_factory=NodeConfig)

    def __post_init__(self):
        self.enode_config.enable_encode = True
        self.enode_config.enable_prefill = False
        self.enode_config.enable_decode = False

        self.pnode_config.enable_encode = False
        self.pnode_config.enable_prefill = True
        self.pnode_config.enable_decode = True
        self.pnode_config.zmq_send_url = self.zmq_url
        self.enode_config.request_processor_config.ep_migrate = True

        self.pnode_config.update_config_value()
        self.enode_config.update_config_value()

    @staticmethod
    def add_curr_config_cli_args(cls, parser: argparse.ArgumentParser, prefix: str="--") -> argparse.ArgumentParser:
        return parser


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
        )(AsyncEPDNode).remote(self.config.enode_config)
        self.pdnode = ray.remote(
            num_cpus=0,
            num_gpus=0.1,
            max_restarts=1,
            name=f'pdnode',
            namespace='dxz',
            lifetime='detached'
        )(AsyncEPDNode).remote(self.config.pnode_config)
        self.nodes = [self.enode, self.pdnode]
        obj1 = self.enode.register_node.remote(self.pdnode)
        ray.get(obj1)
        for node in self.nodes:
            node.step_loop.remote()
            

    async def generate(self, requests: list[Request]) -> list[OfflineInferenceOutput]:
        finished = Counter()
        for request in requests:
            self.enode.add_request.remote(request, RequestProcessParameters(zmq_output=True))
        outputs: list[OfflineInferenceOutput] = []
        while finished.value() < len(requests):
            output = await self.zmq_recv.recv_pyobj()
            outputs.append(output)
            finished.count()
        return outputs


async def main(config):
    print(config)
    requests = [
        Request(
            request_id = i, 
            prompt = f"<image>What is the content of this image?", 
            image = Image.open('./benchmark/dataset/cherry_blossom.jpg'), 
            image_base64 = None, 
            sampling_params = SamplingParameters(max_tokens=10)
        )
        for i in range(8)
    ]
    entrypoint = OfflineEPDDisaggregateEntryPoint(config) 
    outputs = await entrypoint.generate(requests=requests)
    for i, output in enumerate(outputs):
        print(f'output{i}: {output.text}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="offlne epd disaggregate test", conflict_handler='resolve')
    parser = OfflineEPDDisaggregateEntryPointConfig.add_cli_args(parser)
    args = parser.parse_args()
    config = OfflineEPDDisaggregateEntryPointConfig.from_cli_args(args)
    asyncio.run(main(config))