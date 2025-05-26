import ray
from dataclasses import dataclass, field
from typing import Optional, Literal
from dxz.request import Request
from dxz.engine import RequestProcessParameters
from dxz.cluster import NodeConfig, AsyncEPDNode, LoadBalancer, LoadBalancerConfig, MigrateGraphBuilder, MigrateGraph, NodeContext
from dxz.utils.zmq_utils import ZMQConfig
from dxz.utils.socket_utils import parse_port
from dxz.utils.allocate import IncreaingAllocator
from dxz.utils.ray_utils import start_head_node
from dxz.utils.logger import getLogger
logger = getLogger(__name__)


@dataclass
class ClusterConfig:
    zmq: ZMQConfig
    ebalancer: LoadBalancerConfig = field(default_factory= LoadBalancerConfig)
    pbalancer: LoadBalancerConfig = field(default_factory= LoadBalancerConfig)
    cluster_type: Literal['general', 'hybrid', 'single'] = 'hybrid'
    enode: Optional[NodeConfig] = None
    epnode: Optional[NodeConfig] = None
    ednode: Optional[NodeConfig] = None
    epdnode: Optional[NodeConfig] = None
    pnode: Optional[NodeConfig] = None
    pdnode: Optional[NodeConfig] = None
    dnode: Optional[NodeConfig] = None
    n_enode: int = 1
    n_epnode: int = 1
    n_ednode: int = 1
    n_epdnode: int = 1
    n_pnode: int = 1
    n_pdnode: int = 1
    n_dnode: int = 1
    ray_cluster_port: int = 6379
    debug: bool = False

class Cluster:
    def __init__(self, config: ClusterConfig):
        if config.debug:
            config.ray_cluster_port = parse_port(config.ray_cluster_port)
            start_head_node(ray_cluster_port=config.ray_cluster_port)
            ray.init(
                runtime_env={
                    "env_vars": {"RAY_DEBUG_POST_MORTEM": "1"},
                }
            )

        self.config = config
        nodes_list = [
            (getattr(self.config, "n_enode", 0),   getattr(self.config, "enode", None), "e"), 
            (getattr(self.config, "n_epnode", 0),  getattr(self.config, "epnode", None), "ep"), 
            (getattr(self.config, "n_ednode", 0),  getattr(self.config, "ednode", None), "ed"), 
            (getattr(self.config, "n_epdnode", 0), getattr(self.config, "epdnode", None), "epd"), 
            (getattr(self.config, "n_pnode", 0),   getattr(self.config, "pnode", None), "p"), 
            (getattr(self.config, "n_pdnode", 0),  getattr(self.config, "pdnode", None), "pd"), 
            (getattr(self.config, "n_dnode", 0),   getattr(self.config, "dnode", None), "d"), 
        ]

        self.nodes: AsyncEPDNode = []
        self.ebalancer = LoadBalancer(config.ebalancer)
        self.pbalancer = LoadBalancer(config.pbalancer)
        graph_builder = MigrateGraphBuilder()

        world_size = sum([n_node for n_node, _, _ in nodes_list])
        rank_allocator = IncreaingAllocator()
        has_encode: bool = False
        has_prefill: bool = False
        has_decode: bool = False
        for replicas, node_config, node_type in nodes_list:
            for i in range(replicas):
                node = ray.remote(
                    num_cpus=0,
                    num_gpus=1,
                    max_restarts=1,
                    name=f"{node_type}{i}",
                    namespace='dxz',
                    lifetime='detached'
                )(AsyncEPDNode).remote(node_config, NodeContext(
                    rank = rank_allocator.allocate(), 
                    world_size = world_size, 
                    is_ray_actor=True,
                ))
                self.nodes.append(node)
                for ty in node_type:
                    graph_builder.add_node(
                        actor = node, 
                        tpot_slo = node_config.batch_scheduler_profiler.tpot_slo, 
                        node_type = ty,
                    )
                if "e" in node_type:
                    self.ebalancer.register_worker(node)
                    has_encode = True
                if "p" in node_type:
                    has_prefill = True
                    self.pbalancer.register_worker(node)
                if "d" in node_type:
                    has_decode = True

        assert has_prefill and has_decode, "node type is not enough to inference"

        migrate_graph: MigrateGraph = graph_builder.build_graph()
        logger.info(f'{migrate_graph}')

        objs = []
        for node in self.nodes:
            obj = node.init.remote()
            objs.append(obj)
        ray.get(objs)

        objs = []
        for node in self.nodes:
            obj = node.register_migrate_graph.remote(migrate_graph)
            objs.append(obj)
        ray.get(objs)

        for node in self.nodes:
            node.loop.remote()

    def add_request(self, request: Request, params: RequestProcessParameters):
        has_image: bool = request.image is not None or request.image_base64 is not None
        if has_image:
            node = self.ebalancer.choice()
        else:
            node = self.pbalancer.choice()
        node.add_request.remote(request, params)