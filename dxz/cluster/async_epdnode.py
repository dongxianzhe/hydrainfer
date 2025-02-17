import ray
import time
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
from dxz.utils.zmq_utils import init_zmq_send
from dxz.cluster.node_config import NodeConfig
from dxz.cluster.epdnode import EPDNode

from dxz.request.offline_inference_output import OfflineInferenceOutput

class AsyncEPDNode(AsyncEngine):
    def __init__(self, config: NodeConfig):
        self.config = config
        model_factory = getModelFactory(self.config.model_factory_config, ModelFactoryContext())
        self.tokenizer = model_factory.getTokenizer()
        if config.has_kv_cache:
            self.kv_cache_block_manager = TokenCacheBlockManager(config.kv_cache_config, TokenCacheBlockManagerContext())
        else:
            self.kv_cache_block_manager = None

        if config.has_image_cache:
            self.image_cache_block_manager = TokenCacheBlockManager(config.image_cache_config, TokenCacheBlockManagerContext())
        else:
            self.image_cache_block_manager = None

        self.batch_scheduler = BatchScheduler(config.batch_scheduler_config)
        self.worker = getWorker(
            config.worker_config, 
            WorkerContext()
        )
        self.executor = InstructionExecutor(
            config.executor_config, 
            ExecutorContext(
                kv_cache_block_manager = self.kv_cache_block_manager, 
                image_cache_block_manager = self.image_cache_block_manager, 
                worker = self.worker
            )
        )
        self.request_processor = RequestProcessor(
            config.request_processor_config, 
            RequestProcessorContext(
                batch_scheduler=self.batch_scheduler, 
            )
        )
        self.nodes = []
        if config.zmq_send_url:
            self.zmq_send = init_zmq_send(config.zmq_send_url)

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
        for future in futures:
            future.get()
        await self._execute_batch_migrate(batch_migrate)

        # 3. scheduler requests
        t = time.perf_counter()
        for batch in [batch_image_embed, batch_fill, batch_empty]:
            for rcb, inst in batch:
                if rcb.is_finished():
                    rcb.metric.finished_time = t
                    if rcb.virtual_kv_cache: 
                        self.kv_cache_block_manager.realloc(rcb.virtual_kv_cache, 0)
                    if rcb.virtual_image_cache:
                        self.image_cache_block_manager.realloc(rcb.virtual_image_cache, 0)
                    if self.config.zmq_send_url:
                        self.zmq_send.send_pyobj(OfflineInferenceOutput(
                        text = self.tokenizer.decode(rcb.zmq_output.output_token_ids),
                        output_token_ids = rcb.zmq_output.output_token_ids, 
                        arrival_time  = rcb.zmq_output.arrival_time, 
                        finished_time = rcb.zmq_output.finished_time, 
                        token_times = rcb.zmq_output.token_times,
                        ttft = rcb.zmq_output.ttft, 
                        tpot = rcb.zmq_output.tpot, 
                    ))
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
            obj = node.migrate.remote(rcb)