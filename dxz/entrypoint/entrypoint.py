import ray
import asyncio
import uvicorn
import argparse
from typing import Literal
from dataclasses import dataclass, field, fields
from fastapi import FastAPI, Request, Response
from contextlib import asynccontextmanager
from dxz.model import ModelFactoryConfig
from dxz.engine import RequestProcessParameters, OutputTokenParams
from dxz.cluster import NodeConfig, AsyncEPDNode, LoadBalancer, LoadBalancerConfig
from dxz.request import OfflineInferenceOutput
from dxz.utils.config_util import CLIConfig
from dxz.utils.zmq_utils import init_zmq_recv
from dxz.utils.ray_utils import get_ip_address
from dxz.utils.counter import Counter


class APIServer:
    def __init__(self, lifespan=None):
        self.app = FastAPI(lifespan=lifespan)
        self.register_routes()

    def register_routes(self):
        @self.app.get('/health')
        async def health() -> Response:
            return Response(status_code=200)

    def run(self, host: str="127.0.0.1", port: int=8000):
        uvicorn.run(self.app, host=host, port=port, log_level='info')
        

@dataclass
class ClusterConfig(CLIConfig):
    zmq_url: str = f"tcp://{get_ip_address()}:40832"
    model_factory_config: ModelFactoryConfig = field(default_factory=ModelFactoryConfig)
    ednode_config: NodeConfig = field(default_factory=NodeConfig)
    pdnode_config: NodeConfig = field(default_factory=NodeConfig)
    dnode_config: NodeConfig = field(default_factory=NodeConfig)
    n_ednode: int = 1
    n_pdnode: int = 1
    n_dnode: int = 1

    def __post_init__(self):
        self.ednode_config.model_factory_config = self.model_factory_config
        self.pdnode_config.model_factory_config = self.model_factory_config
        self.dnode_config.model_factory_config = self.model_factory_config
        self.ednode_config.update_shared_config()
        self.pdnode_config.update_shared_config()
        self.dnode_config.update_shared_config()

        self.ednode_config.enable_encode = True
        self.ednode_config.enable_prefill = False
        self.ednode_config.enable_decode = True
        self.ednode_config.zmq_send_url = self.zmq_url
        self.ednode_config.request_processor_config.ep_migrate = True
        self.ednode_config.request_processor_config.pd_migrate = True

        self.pdnode_config.enable_encode = False
        self.pdnode_config.enable_prefill = True
        self.pdnode_config.enable_decode = True
        self.pdnode_config.zmq_send_url = self.zmq_url
        self.pdnode_config.request_processor_config.ep_migrate = True
        self.pdnode_config.request_processor_config.pd_migrate = True

        self.dnode_config.enable_encode = False
        self.dnode_config.enable_prefill = False
        self.dnode_config.enable_decode = True
        self.dnode_config.zmq_send_url = self.zmq_url

        self.ednode_config.update_config_value()
        self.pdnode_config.update_config_value()
        self.dnode_config.update_config_value()

    @staticmethod
    def sub_configs_from_cli_args(cls, args: argparse.Namespace, prefix: str="") -> dict[str, 'CLIConfig']:
        return {
            'model_factory_config': ModelFactoryConfig.from_cli_args(args, prefix=prefix), 
            'ednode_config': NodeConfig.from_cli_args(args, prefix=prefix), 
            'pdnode_config': NodeConfig.from_cli_args(args, prefix=prefix), 
            'dnode_config': NodeConfig.from_cli_args(args, prefix=prefix)
        }

    @staticmethod
    def add_sub_configs_cli_args(cls, parser: argparse.ArgumentParser, prefix: str="--") -> argparse.ArgumentParser:
        parser = ModelFactoryConfig.add_cli_args(parser, prefix=prefix)
        parser = NodeConfig.add_cli_args(parser, prefix=prefix+'ed-')
        parser = NodeConfig.add_cli_args(parser, prefix=prefix+'pd-')
        parser = NodeConfig.add_cli_args(parser, prefix=prefix+'d-')
        return parser

    @staticmethod
    def add_curr_config_cli_args(cls, parser: argparse.ArgumentParser, prefix: str="--") -> argparse.ArgumentParser:
        parser.add_argument(f'{prefix}n-ednode', type=int, default=1, help='number of ednodes')
        parser.add_argument(f'{prefix}n-pdnode', type=int, default=1, help='number of pdnodes')
        parser.add_argument(f'{prefix}n-dnode', type=int, default=1, help='number of dnodes')
        return parser


class Cluster:
    def __init__(self, config: ClusterConfig):
        self.config = config
        self.zmq_recv = init_zmq_recv(config.zmq_url)
        self.ednodes = self._create_ray_nodes(config.n_ednode, 'ednode', config.ednode_config)
        self.pdnodes = self._create_ray_nodes(config.n_pdnode, 'pdnode', config.pdnode_config)
        self.dnodes = self._create_ray_nodes(config.n_dnode, 'dnode', config.dnode_config)
        self._register_migrate(self.ednodes, self.pdnodes)
        self._register_migrate(self.pdnodes, self.ednodes)
        self._register_migrate(self.pdnodes, self.dnodes)
        self.ebalancer = LoadBalancer(LoadBalancerConfig(), self.ednodes)
        self.pbalancer = LoadBalancer(LoadBalancerConfig(), self.pdnodes)
        self._start(self.ednodes + self.pdnodes + self.dnodes)

    def get_zmq_recv(self):
        return self.zmq_recv

    def add_request(self, request: Request, params: RequestProcessParameters):
        has_image: bool = request.image is not None or len(request.image_base64) > 0
        if has_image:
            node = self.ebalancer.choice()
        else:
            node = self.pbalancer.choice()
        node.add_request.remote(request, params)

    def _start(self, nodes: list[AsyncEPDNode]):
        for node in nodes:
            node.step_loop.remote()

    def _register_migrate(self, src: list[AsyncEPDNode], dst: list[AsyncEPDNode]):
        objs = []
        for i in src:
            for j in dst:
                obj = i.register_node.remote(j)
                objs.append(obj)
        ray.get(objs)

    def _create_ray_nodes(self, number: int, name: str, config: NodeConfig) -> list[AsyncEPDNode]:
        nodes: list[AsyncEPDNode] = []
        for i in range(number):
            node = ray.remote(
                num_cpus=0,
                num_gpus=1,
                max_restarts=1,
                name=name,
                namespace='dxz',
                lifetime='detached'
            )(AsyncEPDNode).remote(config)
            nodes.append(node)
        return nodes


@dataclass
class EntryPointConfig(CLIConfig):
    host: str = '127.0.0.1'
    port: int = 8888
    cluster_config: ClusterConfig = field(default_factory=ClusterConfig)


class EntryPoint:
    def __init__(self, config: EntryPointConfig):
        self.config = config
        self.api_server = APIServer()
        self.cluster = Cluster(config.cluster_config)
        self.zmq_recv = self.cluster.get_zmq_recv()

    async def generate(self, requests: list[Request]) -> list[OfflineInferenceOutput]:
        finished = Counter()
        for request in requests:
            self.cluster.add_request(request, RequestProcessParameters(
                outout_token_parmas = OutputTokenParams(
                    print_output_text = False, 
                    is_stream_output = False, 
                    is_offline_output = True
                )
            ))
        outputs: list[OfflineInferenceOutput] = []
        while finished.value() < len(requests):
            output = await self.zmq_recv.recv_pyobj()
            outputs.append(output)
            finished.count()
        return outputs


if __name__ == '__main__':
    from PIL import Image
    from dxz.request import SamplingParameters
    from dxz.request import Request
    requests = [
        Request(
            request_id = i, 
            prompt = f"<image>What is the content of this image?", 
            image = Image.open('./benchmark/dataset/cherry_blossom.jpg'), 
            image_base64 = None, 
            sampling_params = SamplingParameters(max_tokens=100)
        )
        for i in range(2)
    ]
    parser = argparse.ArgumentParser(description="entrypoint", conflict_handler='resolve')
    parser = EntryPointConfig.add_cli_args(parser)
    args = parser.parse_args()
    print(f'args {args}')
    config = EntryPointConfig.from_cli_args(args)
    print(f'config {config}')
    entrypoint = EntryPoint(config) 
    outputs = asyncio.run(entrypoint.generate(requests=requests))
    for i, output in enumerate(outputs):
        print(f'output{i}: {output.text}')