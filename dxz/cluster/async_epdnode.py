import ray
import time
import torch
import asyncio
import argparse
from typing import Optional
from dataclasses import dataclass, field, fields
from dxz.request import Request
from dxz.model import ModelFactoryConfig, getModelFactory, ModelFactoryContext
from dxz.engine import AsyncEngine, RequestProcessParameters, RequestControlBlock, BatchRequest, getWorker, BatchScheduler, RequestProcessor, WorkerContext, InstructionExecutor, ExecutorContext, RequestProcessorContext
from dxz.engine import Fill, TextFill, ImageFill, ImageEmbedFill, EmptyInstruction, ImageEmbed, MigrateRequest
from dxz.engine import RequestProcessorConfig, BatchSchedulerConfig, ExecutorConfig, WorkerConfig
from dxz.memory import VirtualTokenCache, TokenCacheBlockManager, TokenCacheBlockManagerConfig, TokenCacheBlockManagerContext
from dxz.engine.output_token_processor import ZmqOutputTokenProcessor
from dxz.engine import BatchSchedulerProfiler, BatchSchedulerConfig, BatchSchedulerProfilerContext
from dxz.utils.zmq_utils import init_zmq_send
from dxz.cluster.node_config import NodeConfig
from dxz.cluster.epdnode import EPDNode
from dxz.cluster.loadbalancer import LoadBalancer, LoadBalancerConfig

from dxz.request.offline_inference_output import OfflineInferenceOutput

class AsyncEPDNode(AsyncEngine):
    def __init__(self, config: NodeConfig):
        self.config = config

        # the name is used in __repr__ which is used to log actor names
        self.name = ""
        if self.config.enable_encode:
            self.name += "E"
        if self.config.enable_prefill:
            self.name += "P"
        if self.config.enable_decode:
            self.name += "D"
        self.name += "Node"

        self.has_vision_model = self.config.enable_encode
        self.has_language_model = self.config.enable_prefill or self.config.enable_decode
        self.has_kv_cache = self.config.enable_prefill or self.config.enable_decode
        self.has_image_cache = self.config.enable_encode or self.config.enable_prefill 

        self.zmq_send = init_zmq_send(config.zmq) if config.zmq else None

        model_factory = getModelFactory(self.config.model, ModelFactoryContext())
        self.tokenizer = model_factory.getTokenizer()
        self.kv_cache_block_manager = TokenCacheBlockManager(config.kv_cache, TokenCacheBlockManagerContext()) if self.has_kv_cache else None
        self.image_cache_block_manager = TokenCacheBlockManager(config.image_cache, TokenCacheBlockManagerContext()) if self.has_image_cache else None
        self.batch_scheduler = BatchScheduler(config.batch_scheduler)
        self.worker = getWorker(config.worker, WorkerContext())
        self.executor = InstructionExecutor(
            config.executor, 
            ExecutorContext(
                kv_cache_block_manager = self.kv_cache_block_manager, 
                image_cache_block_manager = self.image_cache_block_manager, 
                worker = self.worker, 
                zmq_send = self.zmq_send
            )
        )
        self.request_processor = RequestProcessor(
            config.request_processor, 
            RequestProcessorContext(
                batch_scheduler=self.batch_scheduler, 
            )
        )

        self.slow_nodes = []
        self.fast_nodes = []
        self.slow_slo_migrate_scheduler: LoadBalancer = LoadBalancer(LoadBalancerConfig(), self.slow_nodes)
        self.fast_slo_migrate_scheduler: LoadBalancer = LoadBalancer(LoadBalancerConfig(), self.fast_nodes)

        self.profiler = BatchSchedulerProfiler(config.batch_scheduler_profiler, BatchSchedulerProfilerContext(executor=self.executor, kv_cache_block_manager=self.kv_cache_block_manager, image_cache_block_manager=self.image_cache_block_manager))
        if self.config.batch_scheduler_profiler.profile_batch_config:
            if self.config.enable_encode:
                image_budgets = self.profiler.profile_image_budgets()
                self.config.batch_scheduler.max_batch_embed_images = image_budgets

            if (self.config.enable_prefill or self.config.enable_decode):
                token_budgets = self.profiler.profile_token_budgets()
                self.config.batch_scheduler.max_batch_fill_tokens = token_budgets


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

    async def register_node(self, node: "AsyncEngine", fast: bool=True, slow: bool=True): 
        if fast:
            self.fast_slo_migrate_scheduler.register_worker(node)
        if slow:
            self.slow_slo_migrate_scheduler.register_worker(node)

    async def _migrate_virtual_cache(self, virtual_cache: VirtualTokenCache, block_manager: TokenCacheBlockManager) -> VirtualTokenCache:
        new_virtual_cache = block_manager.allocate_virtual_cache()
        block_manager.realloc(new_virtual_cache, virtual_cache.n_cache_tokens)
        block_manager.migrate_blocks(src_virtual_cache=virtual_cache, dst_virtual_cache=new_virtual_cache)
        return new_virtual_cache

    async def migrate(self, rcb: RequestControlBlock):
        if self.config.debug_migrate:
            print(f' migrate request {rcb.request_id} {rcb.instructions}')
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
        if len(contexts) == 0:
            return
        for rcb, _ in contexts:
            rcb.step()
            if rcb.slo_stringent:
                node = self.fast_slo_migrate_scheduler.choice()
            else:
                node = self.slow_slo_migrate_scheduler.choice()
            obj = node.migrate.remote(rcb)
            asyncio.create_task(self._free_migrate_request(rcb))

    async def _free_migrate_request(self, rcb: RequestControlBlock):
        await asyncio.sleep(1)
        await self._free_cache(rcb)

    async def _free_cache(self, rcb: RequestControlBlock):
        if rcb.virtual_kv_cache: 
            self.kv_cache_block_manager.realloc(rcb.virtual_kv_cache, 0)
        if rcb.virtual_image_cache:
            self.image_cache_block_manager.realloc(rcb.virtual_image_cache, 0)

    def __repr__(self):
        return self.name