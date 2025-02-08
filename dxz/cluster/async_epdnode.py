import time
import asyncio
from typing import Optional
from dataclasses import dataclass, field, fields
from dxz.request import Request
from dxz.model import ModelFactoryConfig
from dxz.engine import AsyncEngine, RequestProcessParameters, RequestControlBlock, BatchRequest
from dxz.engine import Fill, TextFill, ImageFill, ImageEmbedFill, EmptyInstruction, ImageEmbed, MigrateRequest
from dxz.engine import RequestProcessorConfig, BatchSchedulerConfig, ExecutorConfig, WorkerConfig, EngineComponentFactory
from dxz.memory import VirtualTokenCache, TokenCacheBlockManager
from dxz.utils.zmq_utils import init_zmq_send


class AsyncEPDNode(AsyncEngine):
    def __init__(
        self, 
        request_processor_config: RequestProcessorConfig, 
        model_factory_config: ModelFactoryConfig, 
        batch_scheduler_config: BatchSchedulerConfig, 
        executor_config: ExecutorConfig, 
        worker_config: WorkerConfig, 
        n_kv_blocks: int, 
        n_image_blocks: int, 
        enable_encode: bool = True, 
        enable_prefill: bool = True, 
        enable_decode: bool = True, 
        zmq_send_url: Optional[str] = None
    ):
        self.enable_encode = enable_encode
        self.enable_prefill = enable_prefill
        self.enable_decode = enable_decode
        self.zmq_send_url = zmq_send_url
        factory = EngineComponentFactory(
            request_processor_config = request_processor_config, 
            model_factory_config     = model_factory_config    , 
            batch_scheduler_config   = batch_scheduler_config  , 
            executor_config          = executor_config         , 
            worker_config            = worker_config           , 
            n_kv_blocks              = n_kv_blocks             , 
            n_image_blocks           = n_image_blocks          , 
        )
        if enable_prefill or enable_decode:
            self.kv_cache_block_manager = factory.get_kv_cache_block_manager()
        else:
            self.kv_cache_block_manager = None

        if enable_encode or enable_prefill:
            self.image_cache_block_manager = factory.get_image_cache_block_manager()
        else:
            self.image_cache_block_manager = None
        self.batch_scheduler = factory.get_batch_scheduler()
        self.executor = factory.get_executor(self.kv_cache_block_manager, self.image_cache_block_manager)
        self.request_processor = factory.get_request_processor(self.batch_scheduler)

        if zmq_send_url:
            self.zmq_send = init_zmq_send(zmq_send_url)

        self.nodes = []

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

        self.executor.execute_image_embed(batch_image_embed)
        self.executor.execute_fill(batch_fill)
        self.executor.execute_empty(batch_empty)
        await self._execute_batch_migrate(batch_migrate)

        # 3. scheduler requests
        t = time.perf_counter()
        for rcb, _ in batch:
            if rcb.is_finished():
                rcb.metric.finished_time = t
                if rcb.virtual_kv_cache: 
                    self.kv_cache_block_manager.realloc(rcb.virtual_kv_cache, 0)
                if rcb.virtual_image_cache:
                    self.image_cache_block_manager.realloc(rcb.virtual_image_cache, 0)
            elif isinstance(inst, MigrateRequest):
                pass
            else:
                self.batch_scheduler.schedule_running(rcb)

    async def step_loop(self):
        while True:
            await self.step()
            await asyncio.sleep(0.001)

    async def register_node(self, node: "AsyncEngine"): 
        self.nodes.append(node)

    async def _migrate_virtual_cache(self, virtual_cache: VirtualTokenCache, block_manager: TokenCacheBlockManager) -> VirtualTokenCache:
        new_virtual_cache = block_manager.allocate_virtual_cache()
        block_manager.realloc(new_virtual_cache, virtual_cache.n_cache_tokens)
        block_manager.migrate_blocks(src_virtual_cache=virtual_cache, dst_virtual_cache=new_virtual_cache)
        return new_virtual_cache

    async def migrate(self, rcb: RequestControlBlock):
        print('migrate is called')
        print(f'rcb.instructions {rcb.instructions}')
        if rcb.virtual_kv_cache:
            rcb.virtual_kv_cache = await self._migrate_virtual_cache(rcb.virtual_kv_cache, self.kv_cache_block_manager) 
        if rcb.virtual_image_cache:
            rcb.virtual_image_cache = await self._migrate_virtual_cache(rcb.virtual_image_cache, self.image_cache_block_manager) 
        print('migrate finished')
        print(f'rcb.virtual_image_cache {rcb.virtual_image_cache.block_table}')
        print(f'rcb.virtual_kv_cache {rcb.virtual_kv_cache}')
        self.batch_scheduler.schedule_new(rcb)
    
    async def _execute_batch_migrate(self, contexts: BatchRequest):
        if len(contexts) == 0:
            return
        node = self.nodes[0]
        for rcb, _ in contexts:
            rcb.step()
            node.migrate.remote(rcb)