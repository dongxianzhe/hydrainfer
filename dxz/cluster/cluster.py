import ray
import argparse
from dataclasses import dataclass, field
from dxz.model import ModelFactoryConfig
from dxz.request import Request
from dxz.engine import RequestProcessParameters
from dxz.cluster import NodeConfig, AsyncEPDNode, LoadBalancer, LoadBalancerConfig
from dxz.utils.ray_utils import get_ip_address
from dxz.utils.zmq_utils import init_zmq_recv, ZMQConfig


@dataclass
class ClusterConfig:
    zmq: ZMQConfig
    model: ModelFactoryConfig = field(default_factory=ModelFactoryConfig)
    ednode: NodeConfig = field(default_factory=NodeConfig)
    pdnode: NodeConfig = field(default_factory=NodeConfig)
    dnode: NodeConfig = field(default_factory=NodeConfig)
    n_ednode: int = 1
    n_pdnode: int = 1
    n_dnode: int = 1

class Cluster:
    def __init__(self, config: ClusterConfig):
        self.config = config
        self.ednodes = self._create_ray_nodes(config.n_ednode, 'ednode', config.ednode)
        self.pdnodes = self._create_ray_nodes(config.n_pdnode, 'pdnode', config.pdnode)
        self.dnodes = self._create_ray_nodes(config.n_dnode, 'dnode', config.dnode)
        self._register_migrate(self.ednodes, self.pdnodes, fast=True, slow=True)
        self._register_migrate(self.pdnodes, self.ednodes, fast=False, slow=True)
        self._register_migrate(self.pdnodes, self.dnodes, fast=True, slow=False)
        self.ebalancer = LoadBalancer(LoadBalancerConfig(), self.ednodes)
        self.pbalancer = LoadBalancer(LoadBalancerConfig(), self.pdnodes)
        self._start(self.ednodes + self.pdnodes + self.dnodes)

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