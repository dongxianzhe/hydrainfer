import torch
import ray
from dataclasses import dataclass, field
from typing import Optional, Literal
from hydrainfer.request import Request
from hydrainfer.engine import RequestProcessParameters
from hydrainfer.cluster import NodeConfig, AsyncEPDNode, LoadBalancer, LoadBalancerConfig, MigrateGraphBuilder, MigrateGraph, NodeContext, NodeType
from hydrainfer.utils.zmq_utils import ZMQConfig
from hydrainfer.utils.socket_utils import parse_port
from hydrainfer.utils.allocate import IncreaingAllocator
from hydrainfer.utils.ray_utils import start_head_node
from hydrainfer.utils.logger import getLogger
logger = getLogger(__name__)


@dataclass
class ClusterConfig:
    zmq: ZMQConfig
    ebalancer: LoadBalancerConfig = field(default_factory= LoadBalancerConfig)
    pbalancer: LoadBalancerConfig = field(default_factory= LoadBalancerConfig)
    cluster_type: Literal['auto', 'general', 'hybrid', 'single'] = 'hybrid'
    enode: Optional[NodeConfig] = None
    epnode: Optional[NodeConfig] = None
    ednode: Optional[NodeConfig] = None
    epdnode: Optional[NodeConfig] = None
    pnode: Optional[NodeConfig] = None
    pdnode: Optional[NodeConfig] = None
    dnode: Optional[NodeConfig] = None
    n_enode: Optional[int] = 1
    n_epnode: Optional[int] = 1
    n_ednode: Optional[int] = 1
    n_epdnode: Optional[int] = 1
    n_pnode: Optional[int] = 1
    n_pdnode: Optional[int] = 1
    n_dnode: Optional[int] = 1
    ray_cluster_port: int = 6379
    debug: bool = False


class DisaggregationMethodProfiler:
    def __init__(self):
        pass


@dataclass
class InstanceDataParallelConfig:
    node_type: Literal["E", "P", "D", "EP", "ED", "PD", "EPD"]
    node_config: Optional[NodeConfig] = None
    n_replicas: int = 0


class Cluster:
    def __init__(self, config: ClusterConfig):
        self.config = config
        if config.debug:
            config.ray_cluster_port = parse_port(config.ray_cluster_port)
            start_head_node(ray_cluster_port=config.ray_cluster_port)
            ray.init(
                runtime_env={
                    "env_vars": {"RAY_DEBUG_POST_MORTEM": "1"},
                }
            )
        if config.cluster_type == 'auto':
            ray.init(ignore_reinit_error=True)
            resources = ray.cluster_resources()
            num_gpus = int(resources['GPU'])
            logger.info(f'ray cluster has resources {resources}')
            assert num_gpus >= 1, "no gpu is available"
            # todo check available memory and profiler
            if num_gpus == 1:
                config.n_epdnode = 1
            else:
                config.n_epnode = num_gpus // 2
                config.n_dnode = num_gpus - config.n_epnode
            logger.info(f"auto set node n_enode={config.n_enode} n_epnode={config.n_epnode} n_ednode={config.n_ednode} n_epdnode={config.n_epdnode} n_pnode={config.n_pnode} n_pdnode={config.n_pdnode} n_dnode={config.n_dnode}")

        instance_data_parallel_config_list: list[InstanceDataParallelConfig] = [
            InstanceDataParallelConfig(n_replicas=getattr(config, "n_enode", 0),   node_config=getattr(config, "enode", None),   node_type="E"), 
            InstanceDataParallelConfig(n_replicas=getattr(config, "n_epnode", 0),  node_config=getattr(config, "epnode", None),  node_type="EP"), 
            InstanceDataParallelConfig(n_replicas=getattr(config, "n_ednode", 0),  node_config=getattr(config, "ednode", None),  node_type="ED"), 
            InstanceDataParallelConfig(n_replicas=getattr(config, "n_epdnode", 0), node_config=getattr(config, "epdnode", None), node_type="EPD"), 
            InstanceDataParallelConfig(n_replicas=getattr(config, "n_pnode", 0),   node_config=getattr(config, "pnode", None),   node_type="P"), 
            InstanceDataParallelConfig(n_replicas=getattr(config, "n_pdnode", 0),  node_config=getattr(config, "pdnode", None),  node_type="PD"), 
            InstanceDataParallelConfig(n_replicas=getattr(config, "n_dnode", 0),   node_config=getattr(config, "dnode", None),   node_type="D"), 
        ]

        # 1. world_size
        world_size = sum([data_parallel_config.n_replicas for data_parallel_config in instance_data_parallel_config_list])

        # 2. verify disaggregation method is valid
        has_encode: bool = False
        has_prefill: bool = False
        has_decode: bool = False
        for data_parallel_config in instance_data_parallel_config_list:
            if "E" in data_parallel_config.node_type:
                has_encode = True
            if "P" in data_parallel_config.node_type:
                has_prefill = True
            if "D" in data_parallel_config.node_type:
                has_decode = True
        assert has_prefill and has_decode, "node type is not enough to inference for inference only text request"
        if not has_encode and has_prefill and has_decode: 
            logger.info("node type is not enough to inference for inference request with image")

        # 3. create node and build migrate graph
        rank_allocator = IncreaingAllocator()
        self.nodes: list[AsyncEPDNode] = []
        self.node_configs: list[NodeConfig] = []
        self.node_contexts: list[NodeContext] = []
        for data_parallel_config in instance_data_parallel_config_list:
            for i in range(data_parallel_config.n_replicas):
                rank = rank_allocator.allocate()
                context = NodeContext(
                    rank = rank, 
                    world_size = world_size, 
                    node_type = NodeType(data_parallel_config.node_type), 
                )
                self.node_contexts.append(context)
                node = ray.remote(
                    num_cpus=0,
                    num_gpus=1,
                    max_restarts=1,
                    name=f"Node{rank}",
                    namespace='hydrainfer',
                    lifetime='detached'
                )(AsyncEPDNode).remote(data_parallel_config.node_config, context)
                self.nodes.append(node)
                self.node_configs.append(data_parallel_config.node_config)

        nccl_config = ray.get(self.nodes[0].get_nccl_config.remote())
        logger.info(f'nccl config {nccl_config}')
        self._run_nodes_remote_method(nodes=self.nodes, method_name="init", wait=True, nccl_config=nccl_config)

        # 4. update migrate graph
        self.ebalancer = LoadBalancer(config.ebalancer)
        self.pbalancer = LoadBalancer(config.pbalancer)

        graph_builder = MigrateGraphBuilder()
        for node, config, context in zip(self.nodes, self.node_configs, self.node_contexts):
            graph_builder.add_node(
                actor = node, 
                tpot_slo = config.batch_scheduler_profiler.tpot_slo,
                node_type = context.node_type,
            )
            if context.node_type.enable_encode:
                self.ebalancer.register_worker(node)
            if context.node_type.enable_prefill:
                self.pbalancer.register_worker(node)
        migrate_graph: MigrateGraph = graph_builder.build_graph()
        logger.info(f'{migrate_graph}')
        for context in self.node_contexts:
            context.migrate_graph = migrate_graph
        objs = [] 
        for node, context in zip(self.nodes, self.node_contexts):
            obj = node.update.remote(context)
            objs.append(obj)
        ray.get(objs)
        self._run_nodes_remote_method(nodes=self.nodes, method_name="loop", wait=False)

    def _run_nodes_remote_method(self, nodes: list[AsyncEPDNode], method_name: str, wait: bool=True, *args, **kwargs):
        objs = []
        for node in nodes:
            method = getattr(node, method_name)
            obj = method.remote(*args, **kwargs)
            objs.append(obj)
        if wait:
            ray.get(objs)

    def add_request(self, request: Request, params: RequestProcessParameters):
        has_image: bool = request.image is not None or request.image_base64 is not None
        if has_image:
            node = self.ebalancer.choice()
        else:
            node = self.pbalancer.choice()
        node.add_request.remote(request, params)