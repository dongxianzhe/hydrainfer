import ray
import traceback
import copy
import time
import torch
import asyncio
import ray.actor
from typing import Optional
import torch.distributed as dist
from torch.distributed import P2POp, batch_isend_irecv
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from dxz.request import Request
from dxz.model import getModelFactory, ModelFactoryContext
from dxz.engine import AsyncEngine, RequestProcessParameters, RequestControlBlock, BatchRequest, getWorker, BatchScheduler, RequestProcessor, WorkerContext, InstructionExecutor, ExecutorContext, EPMigrate, PDMigrate, RequestProcessorObserver
from dxz.memory import VirtualTokenCache, TokenCacheBlockManager, TokenCacheBlockManagerContext, TokenCacheBlockManagerConfig
from dxz.engine import Fill, TextFill, ImageFill, ImageEmbedFill, EmptyInstruction, ImageEmbed, MigrateRequest, PullCache
from dxz.engine import BatchSchedulerProfiler, BatchSchedulerProfilerContext, BatchSchedulerContext, ScenarioType, log_latency_breakdown

from dxz.utils.zmq_utils import init_zmq_send
from dxz.cluster import MigrateGraph, MigrateNode
from dxz.cluster.node_config import NodeConfig
from dxz.cluster.loadbalancer import LoadBalancer, LoadBalancerConfig, CompositeLoadBlancer


@dataclass
class NodeContext:
    rank: int # each engine has a rank
    world_size: int # number of engines
    is_ray_actor: bool = False


class BatchSchedulerObserver(RequestProcessorObserver):
    def __init__(self, batch_scheduler: BatchScheduler):
        self.batch_scheduler = batch_scheduler

    def update(self, rcb: RequestControlBlock):
        self.batch_scheduler.schedule_new(rcb)

class AsyncEPDNode(AsyncEngine):
    def __init__(self, config: NodeConfig, context: NodeContext):
        self.config = config
        self.context = context
        if context.is_ray_actor:
            self.actor_id = ray.get_runtime_context().get_actor_id()
            self.actor_handle = ray.get_runtime_context().current_actor
        # the name is used in __repr__ which is used to log actor names
        self.name = ""
        if self.config.enable_encode:
            self.name += "E"
        if self.config.enable_prefill:
            self.name += "P"
        if self.config.enable_decode:
            self.name += "D"
        self.name += f"NodeRank{self.context.rank}"

    def _init_nccl(self):
        if not self.config.nccl_communicator:
            return
        if self.context.world_size == 1:
            return
        dist.init_process_group(
            backend="nccl", 
            rank=self.context.rank, 
            world_size=self.context.world_size, 
            init_method=f"tcp://{self.config.nccl_communicator.host}:{self.config.nccl_communicator.port}", 
        )
        print('warm up p2p operation')
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

    def _init_engine(self):
        self.has_vision_model = self.config.enable_encode
        self.has_language_model = self.config.enable_prefill or self.config.enable_decode
        self.has_kv_cache = self.config.enable_prefill or self.config.enable_decode
        self.has_image_cache = self.config.enable_encode or self.config.enable_prefill 
        self.worker = getWorker(self.config.worker, WorkerContext())

        model_factory = getModelFactory(self.config.model, ModelFactoryContext())
        language_config = model_factory.getLanguageModelConfig()
        kv_cache_config = TokenCacheBlockManagerConfig(
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
            n_layers = 1, 
            n_tokens = 1, 
            n_blocks = -1, 
            block_size = 576, 
            n_heads = language_config.n_qo_heads, 
            head_size = language_config.head_dim, 
            dtype = self.config.model.dtype, 
            device = self.config.model.device, 
        )
        if self.config.auto_compute_cache_memory:
            total_memory = torch.cuda.get_device_properties(torch.device('cuda:0')).total_memory
            model_memory = torch.cuda.max_memory_allocated()
            reserved_memory = total_memory - model_memory
            activation_memory_utilization = 0.
            if self.has_vision_model:
                activation_memory_utilization += 0.1
            if self.has_language_model:
                activation_memory_utilization += 0.1
            cache_memory_utilization = 1 - activation_memory_utilization
            if self.has_image_cache and self.has_kv_cache:
                image_cache_memory_utilization = cache_memory_utilization * 0.1
                kv_cache_memory_utilization = cache_memory_utilization * 0.8
            elif self.has_image_cache and not self.has_kv_cache:
                image_cache_memory_utilization = cache_memory_utilization
                kv_cache_memory_utilization = 0.
            elif not self.has_image_cache and self.has_kv_cache:
                image_cache_memory_utilization = 0.
                kv_cache_memory_utilization = cache_memory_utilization
            else:
                raise Exception('no cache pool is allocated')
            kv_cache_memory = int(reserved_memory * kv_cache_memory_utilization)
            image_cache_memory = int(reserved_memory * image_cache_memory_utilization)
            print(f'auto compute cache memory: model_memory {model_memory} kv_cache_memory {kv_cache_memory} image_cache_memory {image_cache_memory}')
            kv_cache_config.n_blocks = TokenCacheBlockManager.compute_n_blocks(kv_cache_config, kv_cache_memory)
            image_cache_config.n_blocks = TokenCacheBlockManager.compute_n_blocks(image_cache_config, image_cache_memory)
            print(f'set kv cache n_blocks to {kv_cache_config.n_blocks} image cache n_blocks to {image_cache_config.n_blocks}')

        self.kv_cache_block_manager = TokenCacheBlockManager(kv_cache_config, TokenCacheBlockManagerContext(rank=self.context.rank)) if self.has_kv_cache else None
        self.image_cache_block_manager = TokenCacheBlockManager(image_cache_config, TokenCacheBlockManagerContext(rank=self.context.rank)) if self.has_image_cache else None
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
            ))

        self.request_processor = RequestProcessor(self.config.request_processor)
        self.request_processor.register_output_observer(BatchSchedulerObserver(self.batch_scheduler))

    def _init_migrate(self):
        self.migrate_graph: Optional[MigrateGraph] = None
        self.ep_loadbalancer = CompositeLoadBlancer()
        self.pd_loadbalancer = CompositeLoadBlancer()
        for scenario_type in range(len(ScenarioType)):
            self.ep_loadbalancer.register_loadbalancer(key=scenario_type, loadbalancer=LoadBalancer(LoadBalancerConfig(policy='round')))
            self.pd_loadbalancer.register_loadbalancer(key=scenario_type, loadbalancer=LoadBalancer(LoadBalancerConfig(policy='round')))

        self.migrate_lock = asyncio.Lock()
        self.migrate_pool = ThreadPoolExecutor(max_workers=1)

    async def init(self):
        print(f'init {self.name} actor_id {getattr(self, "actor_id", None)} rank {self.context.rank} world_size {self.context.world_size}')
        self._init_nccl()
        self._init_zmq()
        self._init_engine()
        self._init_migrate()

    async def register_migrate_graph(self, graph: MigrateGraph):
        assert self.migrate_graph is None
        self.migrate_graph = graph
        for table, loadbalancer in [
            (graph.ep_table.get(self.actor_id, []), self.ep_loadbalancer), 
            (graph.pd_table.get(self.actor_id, []), self.pd_loadbalancer), 
        ]:
            for migrate_node in table:
                if migrate_node.tpot_slo < 0.05:
                    loadbalancer.register_worker(key=ScenarioType.Strict, worker=migrate_node)
                    loadbalancer.register_worker(key=ScenarioType.Relaxed, worker=migrate_node)
                else:
                    loadbalancer.register_worker(key=ScenarioType.Relaxed, worker=migrate_node)
        print(f'ep_loadbalancer {self.ep_loadbalancer}')
        print(f'pd_loadbalancer {self.pd_loadbalancer}')

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

        # latency break down analysis
        if self.config.log_latency_breakdown:
            tmp = []
            for rcb, inst in batch_fill:
                tmp.append((rcb, inst))
                if not isinstance(inst, Fill):
                    continue
                if len(inst.token_ids) > 1: # this is a chunked prefill inst
                    if len(rcb.metric.prefill_execute) == 0: # this is first chunk
                        rcb.metric.prefill_execute.append(time.perf_counter())
                    else: # this is not first chunk
                        pass
                else: # this is decode
                    if len(rcb.metric.decode_execute) == 0: # this is first decode
                        rcb.metric.decode_execute.append(time.perf_counter())
        
            for rcb, _ in batch_image_embed:
                rcb.metric.encode_execute.append(time.perf_counter())


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

        # latency break down analysis
        if self.config.log_latency_breakdown:
            for rcb, inst in tmp:
                if not isinstance(inst, Fill):
                    continue
                if len(inst.token_ids) > 1: # this is prefill
                    if len(rcb.metric.prefill_execute) == 1:
                        rcb.metric.prefill_execute.append(time.perf_counter())
                    else:
                        rcb.metric.prefill_execute[1] = time.perf_counter()
                else: # this is decode
                    if len(rcb.metric.decode_execute) == 1:
                        rcb.metric.decode_execute.append(time.perf_counter())
                    elif len(rcb.metric.decode_execute) > 1:
                        rcb.metric.decode_execute[1] = time.perf_counter()
            
            for rcb, _ in batch_image_embed:
                rcb.metric.encode_execute.append(time.perf_counter())

        # 3. scheduler requests
        t = time.perf_counter()
        for batch in [batch_image_embed, batch_fill, batch_empty, batch_pull_cache]:
            for rcb, inst in batch:
                if rcb.is_finished():
                    rcb.metric.finished_time = t
                    await self._free_cache(rcb)
                    if self.config.log_latency_breakdown:
                        log_latency_breakdown(rcb.metric)
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
            import traceback
            traceback.print_exc()
    
    async def perf_monitor_loop(self):
        while True:
            await asyncio.sleep(10)

    async def pull_virtual_cache(self, src_virtual_cache: VirtualTokenCache, dst_virtual_cache: VirtualTokenCache, is_kv_cache: bool):
        block_manager = self.kv_cache_block_manager if is_kv_cache else self.image_cache_block_manager
        if self.config.debug_migrate:
            print(f"3.1 sender response {'kv' if is_kv_cache else 'image'} cache pull request src block table {src_virtual_cache.rank} {src_virtual_cache.block_table} dst block table {dst_virtual_cache.rank} {dst_virtual_cache.block_table}")
        block_manager.migrate_blocks(src_virtual_cache, dst_virtual_cache, is_send=True, backend='nccl')

    def _migrate_virtual_cache(self, src_node_actor_handle: ray.actor.ActorHandle, src_virtual_cache: VirtualTokenCache, is_kv_cache: bool) -> VirtualTokenCache:
        """3. after receiver allocate memory and get ready, tell sender to send cache and receiver wait to receive cache"""
        block_manager = self.kv_cache_block_manager if is_kv_cache else self.image_cache_block_manager

        dst_virtual_cache = new_virtual_cache = block_manager.allocate_virtual_cache()
        block_manager.realloc(dst_virtual_cache, src_virtual_cache.n_cache_tokens)
        if self.config.intranode_migrate_backend == 'ipc':
            if self.config.debug_migrate:
                print(f"3. receiver pull sender {'kv' if is_kv_cache else 'image'} cache via cuda ipc memory handle")
            block_manager.migrate_blocks(src_virtual_cache=src_virtual_cache, dst_virtual_cache=new_virtual_cache, is_send=False, backend='ipc')
        elif self.config.intranode_migrate_backend == 'nccl':
            if self.config.debug_migrate:
                print(f"3. receiver pull sender {'kv' if is_kv_cache else 'image'} cache via nccl")
            if self.config.debug_migrate:
                print(f"3.2 reciver recv {'kv' if is_kv_cache else 'image'} cache pull request src block table {src_virtual_cache.rank} {src_virtual_cache.block_table} dst block table {dst_virtual_cache.rank} {dst_virtual_cache.block_table}")
            src_node_actor_handle.pull_virtual_cache.remote(
                src_virtual_cache=src_virtual_cache, 
                dst_virtual_cache=dst_virtual_cache, 
                is_kv_cache=is_kv_cache, 
            )
            block_manager.migrate_blocks(src_virtual_cache, dst_virtual_cache, is_send=False, backend='nccl')
        else:
            raise Exception(f'invalid intranode migrate backend {self.config.intranode_migrate_backend}')

        return new_virtual_cache

    async def _execute_pull_cache(self, batch: BatchRequest):
        for rcb, inst in batch:
            if len(rcb.metric.ep_transfer) == 0:
                rcb.metric.ep_transfer.append(time.perf_counter())
            else:
                rcb.metric.pd_transfer.append(time.perf_counter())
            
            old_rcb = copy.copy(rcb)
            if rcb.virtual_kv_cache and self.has_kv_cache:
                rcb.virtual_kv_cache = self._migrate_virtual_cache(inst.src_node_actor_handle, rcb.virtual_kv_cache, is_kv_cache=True) 
            else:
                rcb.virtual_kv_cache = None
            if rcb.virtual_image_cache and self.has_image_cache:
                rcb.virtual_image_cache = self._migrate_virtual_cache(inst.src_node_actor_handle, rcb.virtual_image_cache, is_kv_cache=False) 
            else:
                rcb.virtual_image_cache = None
            inst.src_node_actor_handle.free_migrate_request.remote(old_rcb)
            rcb.step()

            if len(rcb.metric.ep_transfer) == 1:
                rcb.metric.ep_transfer.append(time.perf_counter())
            else:
                rcb.metric.pd_transfer.append(time.perf_counter())


    async def migrate(self, src_node_actor_handle: ray.actor.ActorHandle, rcb: RequestControlBlock):
        """ 2. receiver allocate new cache and migrate blocks and called sender's free method to free blocks"""
        if self.config.debug_migrate:
            print(f'2. recv migrate {rcb.scenario_type} request {rcb.request_id} {rcb.instructions}')

        # set pull cache stage src_node_actor_handle, waiting schedule
        rcb.current_instruction().src_node_actor_handle = src_node_actor_handle
        self.batch_scheduler.schedule_new(rcb)
        # self._migrate(src_node_actor_handle, rcb)
    
    async def _execute_batch_migrate(self, batch: BatchRequest):
        """ 1. sender send block table to receiver"""
        if len(batch) == 0:
            return
        for rcb, inst in batch:
            rcb.step()
            loadbalancer = self.ep_loadbalancer if isinstance(inst, EPMigrate) else self.pd_loadbalancer
            node = loadbalancer.choice(key=rcb.scenario_type)
            if node.id == self.actor_id:
                # if migrate to self, skip migrate and pull cache stage and continue schedule running
                rcb.step()
                self.batch_scheduler.schedule_running(rcb)
                continue
            if self.config.debug_migrate:
                print(f'1. sender {inst.ty} migrate to {node.id}, {rcb.scenario_type} request {rcb.request_id} {rcb.instructions}')
            self.batch_scheduler.migrating_acquire()

            # don't know why some rpc call will failed at pickle
            # we do some retries if it failed and terminate the request
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    obj = node.actor.migrate.remote(self.actor_handle, rcb)
                    break
                except Exception as e:
                    print(f"{rcb.request_id} migrate attempt {attempt + 1} failed")
                    traceback.print_exc()
                    await asyncio.sleep(0.5)
            else:
                print(f"{rcb.request_id} migrate failed after {max_retries} attempts")
                self.free_migrate_request(rcb)
                self.zmq_send.send_pyobj((rcb.request_id, None)) # terminate the request

    async def free_migrate_request(self, rcb: RequestControlBlock):
        """ 4. sender free request"""
        if self.config.debug_migrate:
            print(f'4. sender free request {rcb.request_id}')
        await self._free_cache(rcb)
        self.batch_scheduler.migrating_release()

    async def _free_cache(self, rcb: RequestControlBlock):
        if rcb.virtual_kv_cache: 
            self.kv_cache_block_manager.realloc(rcb.virtual_kv_cache, 0)
        if rcb.virtual_image_cache:
            self.image_cache_block_manager.realloc(rcb.virtual_image_cache, 0)

    def __repr__(self):
        return self.name