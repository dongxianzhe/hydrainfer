import ray
from dataclasses import dataclass, field
from typing import Optional, Literal
from dxz.request import Request
from dxz.engine import RequestProcessParameters
from dxz.cluster import NodeConfig, AsyncEPDNode, LoadBalancer, LoadBalancerConfig, MigrateGraphBuilder, MigrateGraph, NodeContext
from dxz.utils.zmq_utils import ZMQConfig
from dxz.utils.allocate import IncreaingAllocator


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

class Cluster:
    def __init__(self, config: ClusterConfig):
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
                    world_size = world_size
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
                if "p" in node_type:
                    self.pbalancer.register_worker(node)
        migrate_graph: MigrateGraph = graph_builder.build_graph()
        migrate_graph.print()

        objs = []
        for node in self.nodes:
            obj = node.register_migrate_graph.remote(migrate_graph)
            objs.append(obj)
        ray.get(objs)

        for node in self.nodes:
            node.step_loop.remote()

    def add_request(self, request: Request, params: RequestProcessParameters):
        has_image: bool = request.image is not None or request.image_base64 is not None
        if has_image:
            node = self.ebalancer.choice()
        else:
            node = self.pbalancer.choice()
        node.add_request.remote(request, params)