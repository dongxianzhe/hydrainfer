import ray
import socket
import traceback
import copy
import time
import torch
import asyncio
import ray.actor
import torch.distributed as dist
from torch.distributed import P2POp, batch_isend_irecv
from dataclasses import dataclass, field
from hydrainfer.utils.socket_utils import parse_address, find_free_port, get_host
from hydrainfer.request import Request
from hydrainfer.model import getModelFactory, ModelFactoryContext
from hydrainfer.engine import RequestProcessParameters, RequestControlBlock, BatchRequest, getWorker, BatchScheduler, RequestProcessor, WorkerConfig, WorkerContext, InstructionExecutor, ExecutorContext, EPMigrate, PDMigrate, RequestProcessorObserver
from hydrainfer.memory import VirtualTokenCache, TokenCacheBlockManager, TokenCacheBlockManagerContext, TokenCacheBlockManagerConfig
from hydrainfer.engine import Fill, TextFill, ImageEmbedFill, EmptyInstruction, ImageEmbed, MigrateRequest, PullCache
from hydrainfer.engine import BatchSchedulerProfiler, BatchSchedulerProfilerContext, BatchSchedulerContext, ScenarioType
from hydrainfer.utils.zmq_utils import init_zmq_send
from hydrainfer.cluster import MigrateGraph, MigrateNode, NodeType
from hydrainfer.cluster.node_config import NodeConfig, NCCLCommunicatorConfig
from hydrainfer.cluster.loadbalancer import LoadBalancer, LoadBalancerConfig, CompositeLoadBlancer
from hydrainfer.utils.logger import getLogger
logger = getLogger(__name__)

@dataclass
class NodeContext:
    rank: int # each engine has a rank
    world_size: int # number of engines
    node_type: NodeType = field(default_factory=NodeType)
    migrate_graph: MigrateGraph = field(default_factory=MigrateGraph)


class BatchSchedulerObserver(RequestProcessorObserver):
    def __init__(self, batch_scheduler: BatchScheduler):
        self.batch_scheduler = batch_scheduler

    def update(self, rcb: RequestControlBlock):
        self.batch_scheduler.schedule_new(rcb)

class AsyncEPDNode:
    def __init__(self, config: NodeConfig, context: NodeContext):
        self.config = config
        self.context = context
        try:
            self.actor_id = ray.get_runtime_context().get_actor_id()
            self.actor_handle = ray.get_runtime_context().current_actor
        except Exception as e:
            raise Exception("AsyncEPDNode should be used as an ray actor")
        self._update_actor_name(context)

    def __repr__(self):
        return self.name

    def _update_actor_name(self, context: NodeContext):
        # the name is used in __repr__ which is used to log actor names
        self.name = f"{context.node_type.node_type}NodeRank{context.rank}"

    def _update_migrate_graph(self, context: NodeContext):
        self.ep_loadbalancer = CompositeLoadBlancer()
        self.pd_loadbalancer = CompositeLoadBlancer()
        for scenario_type in range(len(ScenarioType)):
            self.ep_loadbalancer.register_loadbalancer(key=scenario_type, loadbalancer=LoadBalancer(LoadBalancerConfig(policy='round')))
            self.pd_loadbalancer.register_loadbalancer(key=scenario_type, loadbalancer=LoadBalancer(LoadBalancerConfig(policy='round')))

        assert context.migrate_graph is not None
        for table, loadbalancer in [
            (context.migrate_graph.ep_table.get(self.actor_id, []), self.ep_loadbalancer), 
            (context.migrate_graph.pd_table.get(self.actor_id, []), self.pd_loadbalancer), 
        ]:
            for migrate_node in table:
                if migrate_node.tpot_slo < 0.05:
                    loadbalancer.register_worker(key=ScenarioType.Strict, worker=migrate_node)
                    loadbalancer.register_worker(key=ScenarioType.Relaxed, worker=migrate_node)
                else:
                    loadbalancer.register_worker(key=ScenarioType.Relaxed, worker=migrate_node)
        logger.info(f'ep_loadbalancer {self.ep_loadbalancer}')
        logger.info(f'pd_loadbalancer {self.pd_loadbalancer}')

    def _update_cache(self, context: NodeContext):
        model_factory = getModelFactory(self.config.model, ModelFactoryContext())
        language_config = model_factory.getLanguageModelConfig()
        kv_cache_config = TokenCacheBlockManagerConfig(
            communication_backend_manager_config=self.config.migrate_backend_config, 
            n_layers = language_config.n_layers, 
            n_tokens = 2, 
            n_blocks = -1, 
            block_size = 16, 
            n_heads = language_config.n_kv_heads, 
            head_size = language_config.head_dim, 
            dtype = self.config.model.dtype, 
            device = self.config.model.device, 
        )
        image_cache_config = TokenCacheBlockManagerConfig(
            communication_backend_manager_config=self.config.migrate_backend_config, 
            n_layers = 1, 
            n_tokens = 1, 
            n_blocks = -1, 
            block_size = 576, 
            n_heads = language_config.n_qo_heads, 
            head_size = language_config.head_dim, 
            dtype = self.config.model.dtype, 
            device = self.config.model.device, 
        )
        total_memory = torch.cuda.get_device_properties(torch.device(self.config.model.device)).total_memory
        model_memory = torch.cuda.max_memory_allocated()
        reserved_memory = total_memory - model_memory
        activation_memory_utilization = 0.
        if context.node_type.has_vision_model:
            activation_memory_utilization += 0.1
        if context.node_type.has_language_model:
            activation_memory_utilization += 0.1
        cache_memory_utilization = 1 - activation_memory_utilization
        if context.node_type.has_image_cache and context.node_type.has_kv_cache:
            image_cache_memory_utilization = cache_memory_utilization * 0.1
            kv_cache_memory_utilization = cache_memory_utilization * 0.8
        elif context.node_type.has_image_cache and not context.node_type.has_kv_cache:
            image_cache_memory_utilization = cache_memory_utilization
            kv_cache_memory_utilization = 0.
        elif not context.node_type.has_image_cache and context.node_type.has_kv_cache:
            image_cache_memory_utilization = 0.
            kv_cache_memory_utilization = cache_memory_utilization
        else:
            raise Exception('no cache pool is allocated')
        kv_cache_memory = int(reserved_memory * kv_cache_memory_utilization)
        image_cache_memory = int(reserved_memory * image_cache_memory_utilization)
        logger.info(f'auto compute cache memory: model_memory {model_memory} kv_cache_memory {kv_cache_memory}B image_cache_memory {image_cache_memory}B')
        kv_cache_config.n_blocks = TokenCacheBlockManager.compute_n_blocks(kv_cache_config, kv_cache_memory)
        image_cache_config.n_blocks = TokenCacheBlockManager.compute_n_blocks(image_cache_config, image_cache_memory)
        logger.info(f'auto set kv cache n_blocks to {kv_cache_config.n_blocks} image cache n_blocks to {image_cache_config.n_blocks}')
        self.kv_cache_block_manager = TokenCacheBlockManager(kv_cache_config, TokenCacheBlockManagerContext(rank=context.rank, rank2host=self.rank2host)) if context.node_type.has_kv_cache else None
        self.image_cache_block_manager = TokenCacheBlockManager(image_cache_config, TokenCacheBlockManagerContext(rank=context.rank, rank2host=self.rank2host)) if context.node_type.has_image_cache else None

        if self.config.batch_scheduler.max_running_requests == -1:
            # suppose avg 2048 token for each request
            max_running_requests = 1024
            if context.node_type.has_kv_cache:
                max_running_requests = min(max_running_requests, kv_cache_config.n_blocks // (2048 // kv_cache_config.block_size))
            if context.node_type.has_image_cache:
                max_running_requests = min(max_running_requests, image_cache_config.n_blocks // (2048 // image_cache_config.block_size))
            self.config.batch_scheduler.max_running_requests = max_running_requests
            logger.info(f'auto set batch scheduler max_running_requests {self.config.batch_scheduler.max_running_requests}')

    def _update_worker(self, context: NodeContext):
        worker_config = WorkerConfig(model=self.config.model)
        self.worker = getWorker(worker_config, WorkerContext(has_vision_model=context.node_type.has_vision_model, has_language_model=context.node_type.has_language_model))

    def _update_engine(self, context: NodeContext):
        self._update_worker(context)
        self._update_cache(context)

        self.executor = InstructionExecutor(
            self.config.executor, 
            ExecutorContext(
                kv_cache_block_manager = self.kv_cache_block_manager, 
                image_cache_block_manager = self.image_cache_block_manager, 
                worker = self.worker, 
                zmq_send = self.zmq_send
            ))

        self.profiler = BatchSchedulerProfiler(
            self.config.batch_scheduler_profiler, 
            BatchSchedulerProfilerContext(
                executor=self.executor, 
                kv_cache_block_manager=self.kv_cache_block_manager, 
                image_cache_block_manager=self.image_cache_block_manager
            ))

        self.batch_scheduler = BatchScheduler(
            self.config.batch_scheduler, 
            BatchSchedulerContext(
                profiler = self.profiler, 
                kv_cache_block_manager=self.kv_cache_block_manager, 
                image_cache_block_manager=self.image_cache_block_manager, 
            ))

        self.request_processor = RequestProcessor(self.config.request_processor)
        self.request_processor.register_output_observer(BatchSchedulerObserver(self.batch_scheduler))

    async def update(self, context: NodeContext):
        try:
            self._update_actor_name(context)
            assert self.context.rank == context.rank, "auto scale and rank change dynamically is not supported"
            assert self.context.world_size == context.world_size, "auto scale and rank change dynamically is not supported"
            if context.node_type != context.node_type:
                self._update_engine(context)
            self._update_migrate_graph(context)
            self.context = context
        except: 
            traceback.print_exc()

    def _init_nccl(self, nccl_config: NCCLCommunicatorConfig):
        if self.context.world_size == 1:
            return
        dist.init_process_group(
            backend="nccl", 
            rank=self.context.rank, 
            world_size=self.context.world_size, 
            init_method=parse_address(nccl_config), 
        )
        logger.info('warm up p2p operation')
        p2p_op_list = []
        if self.context.rank == 0:
            for i in range(1, self.context.world_size):
                tensor = torch.ones(size=(1, ), dtype=torch.int, device=torch.device('cuda:0'))
                p2p_op_list.append(P2POp(dist.isend, tensor, i))
        else:
            tensor = torch.zeros(size=(1, ), dtype=torch.int, device=torch.device('cuda:0'))
            p2p_op_list.append(P2POp(dist.irecv, tensor, 0))
        reqs = batch_isend_irecv(p2p_op_list)
        for req in reqs:
            req.wait()

    def _init_zmq(self):
        self.zmq_send = init_zmq_send(self.config.zmq) if self.config.zmq else None

    async def init(self, nccl_config: NCCLCommunicatorConfig, rank2host: dict[int, str]):
        try:
            self.rank2host = rank2host
            logger.info(f'init {self.name} actor_id {getattr(self, "actor_id", None)} rank {self.context.rank} world_size {self.context.world_size}')
            logger.info("init nccl")
            self._init_nccl(nccl_config)
            logger.info("init zmq")
            self._init_zmq()
            logger.info("init engine")
            self._update_engine(self.context)
            logger.info("init node success")
        except Exception as e:
            traceback.print_exc()

    def get_nccl_config(self) -> NCCLCommunicatorConfig:
        return NCCLCommunicatorConfig(host=get_host(), port=find_free_port())

    def get_rank_host(self) -> tuple[int, str]:
        return self.context.rank, get_host()

    async def add_request(self, request: Request, params: RequestProcessParameters):
        self.request_processor.process(request, params)

    async def step(self):
        # 1. schedule requests
        batch = self.batch_scheduler.step()
        if len(batch) == 0:
            return

        # 2. execute instructions
        batch_fill = BatchRequest()
        batch_image_embed = BatchRequest()
        batch_empty = BatchRequest()
        batch_migrate = BatchRequest()
        batch_pull_cache = BatchRequest() 
        for rcb, inst in batch:
            if isinstance(inst, Fill):
                batch_fill.append(rcb)
                continue
            if isinstance(inst, EmptyInstruction):
                batch_empty.append(rcb)
                continue
            if isinstance(inst, ImageEmbed):
                batch_image_embed.append(rcb)
                continue
            if isinstance(inst, MigrateRequest):
                batch_migrate.append(rcb)
                continue
            if isinstance(inst, PullCache):
                batch_pull_cache.append(rcb)
                continue
            raise Exception(f'unsupported instrction {type(inst)}')


        # execute concurrently
        async_futures = []
        async_futures.append(asyncio.create_task(self._execute_batch_migrate(batch_migrate)))
        async_futures.append(asyncio.create_task(self._execute_pull_cache(batch_pull_cache)))
        futures = []
        futures.append(self.executor.execute_image_embed(batch_image_embed))
        futures.append(self.executor.execute_fill(batch_fill))
        futures.append(self.executor.execute_empty(batch_empty))

        for future in futures:
            future.get()
        asyncio.gather(*async_futures)
        if self.kv_cache_block_manager:
            self.kv_cache_block_manager.synchronize()
        if self.image_cache_block_manager:
            self.image_cache_block_manager.synchronize()

        # 3. scheduler requests
        t = time.perf_counter()
        for batch in [batch_image_embed, batch_fill, batch_empty, batch_pull_cache]:
            for rcb, inst in batch:
                if rcb.is_finished():
                    self.batch_scheduler.free_request(rcb)
                else:
                    self.batch_scheduler.schedule_running(rcb)

    async def loop(self):
        asyncio.gather(
            asyncio.create_task(self.step_loop()), 
            asyncio.create_task(self.perf_monitor_loop()), 
        )

    async def step_loop(self):
        try:
            while True:
                await self.step()
                await asyncio.sleep(0.001)
        except Exception as e:
            traceback.print_exc()
            ray.actor.exit_actor()
    
    async def perf_monitor_loop(self):
        try:
            while True:
                logger.info(
                    f'image cache usage {self.image_cache_block_manager.get_metrics() if self.image_cache_block_manager else ""}, '
                    f'kv cache usage {self.kv_cache_block_manager.get_metrics()if self.kv_cache_block_manager else ""}, '
                    f'{self.batch_scheduler.get_metrics(), }'
                )
                await asyncio.sleep(5)
        except Exception as e:
            traceback.print_exc()
    
    async def _execute_batch_migrate(self, batch: BatchRequest):
        """ 1. sender send block table to receiver"""
        if len(batch) == 0:
            return
        for rcb, inst in batch:
            loadbalancer = self.ep_loadbalancer if isinstance(inst, EPMigrate) else self.pd_loadbalancer
            node = loadbalancer.choice(key=rcb.scenario_type)
            if node.id == self.actor_id:
                # if migrate to self, skip migrate and pull cache stage and continue schedule running
                rcb.step()
                rcb.step()
                self.batch_scheduler.schedule_running(rcb)
                continue
            self.batch_scheduler.schedule_waiting_to_be_pulled(rcb)

            new_rcb = copy.deepcopy(rcb)
            new_rcb.step()
            assert isinstance(new_rcb.current_instruction(), PullCache)
            new_rcb.current_instruction().src_node_actor_handle = self.actor_handle
            new_rcb.current_instruction().virtual_kv_cache = rcb.virtual_kv_cache
            new_rcb.current_instruction().virtual_image_cache = rcb.virtual_image_cache
            new_rcb.virtual_kv_cache = None
            new_rcb.virtual_image_cache = None
            # don't know why some rpc call will failed at pickle
            # we do some retries if it failed and terminate the request
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    obj = node.actor.add_migrate_request.remote(new_rcb)
                    break
                except Exception as e:
                    logger.warning(f"{rcb.request_id} migrate attempt {attempt + 1} failed")
                    traceback.print_exc()
                    await asyncio.sleep(0.5)
            else:
                logger.warning(f"{rcb.request_id} migrate failed after {max_retries} attempts")
                self.batch_scheduler.free_migrated_request(rcb.request_id)
                self.zmq_send.send_pyobj((rcb.request_id, None)) # terminate the request

    async def add_migrate_request(self, rcb: RequestControlBlock):
        """ 2. receiver allocate new cache and waiting scheduler to pull cache"""
        self.batch_scheduler.schedule_waiting_to_pull(rcb)

    async def _execute_pull_cache(self, batch: BatchRequest):
        """3. called sender to send blocks and free blocks"""
        for rcb, inst in batch:
            inst: PullCache
            src_node_actor_handle = inst.src_node_actor_handle
            if inst.virtual_kv_cache and rcb.virtual_kv_cache:
                assert len(inst.virtual_kv_cache.block_table) == len(rcb.virtual_kv_cache.block_table), f'{len(inst.virtual_kv_cache.block_table)} {inst.virtual_kv_cache.n_cache_tokens} {len(rcb.virtual_kv_cache.block_table)} {rcb.virtual_kv_cache.n_cache_tokens}'
                src_node_actor_handle.send_cache.remote(
                    src_virtual_cache=inst.virtual_kv_cache, 
                    dst_virtual_cache=rcb.virtual_kv_cache, 
                    is_kv_cache=True, 
                )
                self.kv_cache_block_manager.migrate_blocks(inst.virtual_kv_cache, rcb.virtual_kv_cache, is_send=False)
            if inst.virtual_image_cache and rcb.virtual_image_cache:
                assert len(inst.virtual_image_cache.block_table) == len(rcb.virtual_image_cache.block_table), f'{len(inst.virtual_image_cache.block_table)} {len(rcb.virtual_image_cache.block_table)} {inst.virtual_image_cache.block_table} {rcb.virtual_image_cache.block_table}'
                src_node_actor_handle.send_cache.remote(
                    src_virtual_cache=inst.virtual_image_cache, 
                    dst_virtual_cache=rcb.virtual_image_cache, 
                    is_kv_cache=False, 
                )
                self.image_cache_block_manager.migrate_blocks(inst.virtual_image_cache, rcb.virtual_image_cache, is_send=False)
            inst.src_node_actor_handle.free_migrated_request.remote(rcb.request_id)
            rcb.step()

    async def send_cache(self, src_virtual_cache: VirtualTokenCache, dst_virtual_cache: VirtualTokenCache, is_kv_cache: bool):
        try:
            block_manager = self.kv_cache_block_manager if is_kv_cache else self.image_cache_block_manager
            block_manager.migrate_blocks(src_virtual_cache, dst_virtual_cache, is_send=True)
        except Exception as e:
            traceback.print_exc()
            # ray.actor.exit_actor()


    async def free_migrated_request(self, request_id: int):
        """ 4. sender free request"""
        try:
            self.batch_scheduler.free_migrated_request(request_id)
            pass
        except Exception as e:
            traceback.print_exc()
            # ray.actor.exit_actor()