import ray
import time
import asyncio
from typing import Optional
from dxz.request import Request
from dxz.model import getModelFactory, ModelFactoryContext
from dxz.engine import AsyncEngine, RequestProcessParameters, RequestControlBlock, BatchRequest, getWorker, BatchScheduler, RequestProcessor, WorkerContext, InstructionExecutor, ExecutorContext, RequestProcessorContext
from dxz.memory import VirtualTokenCache, TokenCacheBlockManager, TokenCacheBlockManagerContext
from dxz.engine import Fill, TextFill, ImageFill, ImageEmbedFill, EmptyInstruction, ImageEmbed, MigrateRequest
from dxz.engine import BatchSchedulerProfiler, BatchSchedulerProfilerContext, BatchSchedulerContext, ScenarioType
from dxz.utils.zmq_utils import init_zmq_send
from dxz.cluster import MigrateGraph, MigrateNode
from dxz.cluster.node_config import NodeConfig
from dxz.cluster.loadbalancer import LoadBalancer, LoadBalancerConfig, CompositeLoadBlancer

class AsyncEPDNode(AsyncEngine):
    def __init__(self, config: NodeConfig):
        self.config = config
        self.actor_id = ray.get_runtime_context().get_actor_id()

        # the name is used in __repr__ which is used to log actor names
        self.name = ""
        if self.config.enable_encode:
            self.name += "E"
        if self.config.enable_prefill:
            self.name += "P"
        if self.config.enable_decode:
            self.name += "D"
        self.name += "Node"
        print(f'start {self.name} actor id {self.actor_id}')

        self.has_vision_model = self.config.enable_encode
        self.has_language_model = self.config.enable_prefill or self.config.enable_decode
        self.has_kv_cache = self.config.enable_prefill or self.config.enable_decode
        self.has_image_cache = self.config.enable_encode or self.config.enable_prefill 

        self.zmq_send = init_zmq_send(config.zmq) if config.zmq else None

        model_factory = getModelFactory(self.config.model, ModelFactoryContext())
        self.tokenizer = model_factory.getTokenizer()

        self.kv_cache_block_manager = TokenCacheBlockManager(config.kv_cache, TokenCacheBlockManagerContext()) if self.has_kv_cache else None
        self.image_cache_block_manager = TokenCacheBlockManager(config.image_cache, TokenCacheBlockManagerContext()) if self.has_image_cache else None

        self.worker = getWorker(config.worker, WorkerContext())
        self.executor = InstructionExecutor(
            config.executor, 
            ExecutorContext(
                kv_cache_block_manager = self.kv_cache_block_manager, 
                image_cache_block_manager = self.image_cache_block_manager, 
                worker = self.worker, 
                zmq_send = self.zmq_send
            ))

        self.profiler = BatchSchedulerProfiler(
            config.batch_scheduler_profiler, 
            BatchSchedulerProfilerContext(
                executor=self.executor, 
                kv_cache_block_manager=self.kv_cache_block_manager, 
                image_cache_block_manager=self.image_cache_block_manager
            ))

        self.batch_scheduler = BatchScheduler(
            config.batch_scheduler, 
            BatchSchedulerContext(
                profiler = self.profiler, 
            ))

        self.request_processor = RequestProcessor(
            config.request_processor, 
            RequestProcessorContext(
                batch_scheduler=self.batch_scheduler, 
            ))

        self.migrate_graph: Optional[MigrateGraph] = None
        self.ep_loadbalancer = CompositeLoadBlancer()
        self.pd_loadbalancer = CompositeLoadBlancer()
        for scenario_type in range(len(ScenarioType)):
            self.ep_loadbalancer.register_loadbalancer(key=scenario_type, loadbalancer=LoadBalancer(LoadBalancerConfig(policy='round')))
            self.pd_loadbalancer.register_loadbalancer(key=scenario_type, loadbalancer=LoadBalancer(LoadBalancerConfig(policy='round')))

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
            raise Exception(f'unsupported instrction {type(inst)}')

        futures = []
        futures.append(self.executor.execute_image_embed(batch_image_embed))
        futures.append(self.executor.execute_fill(batch_fill))
        futures.append(self.executor.execute_empty(batch_empty))
        await self._execute_batch_migrate(batch_migrate)
        for future in futures:
            future.get()

        # 3. scheduler requests
        t = time.perf_counter()
        for batch in [batch_image_embed, batch_fill, batch_empty]:
            for rcb, inst in batch:
                if rcb.is_finished():
                    rcb.metric.finished_time = t
                    await self._free_cache(rcb)
                else:
                    self.batch_scheduler.schedule_running(rcb)

    async def step_loop(self):
        while True:
            await self.step()
            await asyncio.sleep(0.001)

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
                else:
                    loadbalancer.register_worker(key=ScenarioType.Relaxed, worker=migrate_node)
        print(f'self.ep_loadbalancer {self.ep_loadbalancer}')
        print(f'self.pd_loadbalancer {self.pd_loadbalancer}')


    async def _migrate_virtual_cache(self, virtual_cache: VirtualTokenCache, block_manager: TokenCacheBlockManager) -> VirtualTokenCache:
        new_virtual_cache = block_manager.allocate_virtual_cache()
        block_manager.realloc(new_virtual_cache, virtual_cache.n_cache_tokens)
        block_manager.migrate_blocks(src_virtual_cache=virtual_cache, dst_virtual_cache=new_virtual_cache)
        return new_virtual_cache

    async def migrate(self, rcb: RequestControlBlock):
        """ 2. receiver allocate new cache and migrate blocks and called sender's free method to free blocks"""
        if self.config.debug_migrate:
            print(f'recv migrate {rcb.scenario_type} request {rcb.request_id} {rcb.instructions}')
        if rcb.virtual_kv_cache and self.has_kv_cache:
            rcb.virtual_kv_cache = await self._migrate_virtual_cache(rcb.virtual_kv_cache, self.kv_cache_block_manager) 
        else:
            rcb.virtual_kv_cache = None
        if rcb.virtual_image_cache and self.has_image_cache:
            rcb.virtual_image_cache = await self._migrate_virtual_cache(rcb.virtual_image_cache, self.image_cache_block_manager) 
        else:
            rcb.virtual_image_cache = None
        self.batch_scheduler.schedule_new(rcb)
    
    async def _execute_batch_migrate(self, contexts: BatchRequest):
        """ 1. sender called _execute_batch_migrate to call receiver's migrate method"""
        if len(contexts) == 0:
            return
        for rcb, inst in contexts:
            rcb.step()
            loadbalancer = self.ep_loadbalancer if inst.ty == 'ep' else self.pd_loadbalancer
            node = loadbalancer.choice(key=rcb.scenario_type)
            if node.id == self.actor_id:
                # if migrate to self, skip migrate and schedule running
                self.batch_scheduler.schedule_running(rcb)
                continue
            if self.config.debug_migrate:
                print(f'1. sender {inst.ty} migrate to {node.id}, {rcb.scenario_type} request {rcb.request_id} {rcb.instructions}')
            obj = node.actor.migrate.remote(rcb)

            # todo receiver call sender _free_migrate_request to free migrate request
            asyncio.create_task(self.free_migrate_request(rcb))

    async def free_migrate_request(self, rcb: RequestControlBlock):
        if self.config.debug_migrate:
            print(f'3. sender free request {rcb.request_id}')
        await self._free_cache(rcb)

    async def _free_cache(self, rcb: RequestControlBlock):
        if rcb.virtual_kv_cache: 
            self.kv_cache_block_manager.realloc(rcb.virtual_kv_cache, 0)
        if rcb.virtual_image_cache:
            self.image_cache_block_manager.realloc(rcb.virtual_image_cache, 0)

    def __repr__(self):
        return self.name