import ray
import argparse
from dataclasses import dataclass, field
from dxz.model import ModelFactoryConfig
from dxz.request import Request
from dxz.engine import RequestProcessParameters
from dxz.cluster import NodeConfig, AsyncEPDNode, LoadBalancer, LoadBalancerConfig
from dxz.utils.ray_utils import get_ip_address
from dxz.utils.config_util import CLIConfig
from dxz.utils.zmq_utils import init_zmq_recv


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
        self.pdnode_config.request_processor_config.ep_migrate = False
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
        parser = NodeConfig.add_cli_args(parser, prefix=prefix)
        # parser = NodeConfig.add_cli_args(parser, prefix=prefix+'ed-')
        # parser = NodeConfig.add_cli_args(parser, prefix=prefix+'pd-')
        # parser = NodeConfig.add_cli_args(parser, prefix=prefix+'d-')
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
        self._register_migrate(self.ednodes, self.pdnodes, fast=True, slow=True)
        self._register_migrate(self.pdnodes, self.ednodes, fast=True, slow=True)
        self._register_migrate(self.pdnodes, self.dnodes, fast=True, slow=True)
        self.ebalancer = LoadBalancer(LoadBalancerConfig(), self.ednodes)
        self.pbalancer = LoadBalancer(LoadBalancerConfig(), self.pdnodes)
        self._start(self.ednodes + self.pdnodes + self.dnodes)

    def get_zmq_recv(self):
        return self.zmq_recv

    def add_request(self, request: Request, params: RequestProcessParameters):
        has_image: bool = request.image is not None or request.image_base64 is not None
        if has_image:
            node = self.ebalancer.choice()
        else:
            node = self.pbalancer.choice()
        node.add_request.remote(request, params)

    def _start(self, nodes: list[AsyncEPDNode]):
        for node in nodes:
            node.step_loop.remote()

    def _register_migrate(self, src: list[AsyncEPDNode], dst: list[AsyncEPDNode], fast: bool=True, slow: bool=True):
        objs = []
        for i in src:
            for j in dst:
                obj = i.register_node.remote(j, fast=fast, slow=slow)
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