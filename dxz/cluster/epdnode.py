import time
import torch
import argparse
from dataclasses import dataclass, field, fields
from dxz.request import Request
from dxz.model import getModelFactory, ModelFactoryConfig, ModelFactoryContext
from dxz.engine import RequestProcessor, RequestProcessorConfig, ImageEmbed, Fill, EmptyInstruction, BatchScheduler, BatchSchedulerConfig, ExecutorConfig, WorkerConfig, getWorker, WorkerContext, ExecutorContext, InstructionExecutor, Engine, BatchRequest, RequestProcessParameters
from dxz.memory import TokenCacheBlockManager, TokenCacheBlockManagerContext, TokenCacheBlockManagerConfig
from dxz.cluster.node_config import NodeConfig


class EPDNode(Engine):
    def __init__(self, config: NodeConfig):
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

    def add_request(self, request: Request, params: RequestProcessParameters):
        self.request_processor.process(request, params)

    @torch.inference_mode()
    def step(self):
        # 1. schedule requests
        batch = self.batch_scheduler.step()
        if len(batch) == 0:
            return

        # 2. execute instructions
        batch_fill = BatchRequest()
        batch_image_embed = BatchRequest()
        batch_empty = BatchRequest()
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
            raise Exception(f'unsupported instrction {type(inst)}')

        self.executor.execute_image_embed(batch_image_embed)
        self.executor.execute_fill(batch_fill)
        self.executor.execute_empty(batch_empty)

        # 3. scheduler requests
        t = time.perf_counter()
        for rcb, _ in batch:
            if rcb.is_finished():
                rcb.metric.finished_time = t
                if rcb.virtual_kv_cache: 
                    self.kv_cache_block_manager.realloc(rcb.virtual_kv_cache, 0)
                if rcb.virtual_image_cache:
                    self.image_cache_block_manager.realloc(rcb.virtual_image_cache, 0)
            else:
                self.batch_scheduler.schedule_running(rcb)