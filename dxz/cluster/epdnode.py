import time
import torch
import argparse
from dataclasses import dataclass, field, fields
from dxz.request import Request
from dxz.model import getModelFactory, ModelFactoryConfig, ModelFactoryContext
from dxz.engine import RequestProcessor, RequestProcessorConfig, RequestProcessorContext, ImageEmbed, Fill, EmptyInstruction, BatchScheduler, BatchSchedulerConfig, ExecutorConfig, WorkerConfig, getWorker, WorkerContext, ExecutorContext, InstructionExecutor, Engine, BatchRequest, RequestProcessParameters, EngineComponentFactory
from dxz.memory import TokenCacheBlockManager, TokenCacheBlockManagerContext


class EPDNode(Engine):
    def __init__(self, 
            request_processor_config: RequestProcessorConfig, 
            model_factory_config: ModelFactoryConfig, 
            batch_scheduler_config: BatchSchedulerConfig, 
            executor_config: ExecutorConfig, 
            worker_config: WorkerConfig, 
            n_kv_blocks: int, 
            n_image_blocks: int, 
        ):
        factory = EngineComponentFactory(
            request_processor_config = request_processor_config, 
            model_factory_config     = model_factory_config    , 
            batch_scheduler_config   = batch_scheduler_config  , 
            executor_config          = executor_config         , 
            worker_config            = worker_config           , 
            n_kv_blocks              = n_kv_blocks             , 
            n_image_blocks           = n_image_blocks          , 
        )
        self.kv_cache_block_manager = factory.get_kv_cache_block_manager()
        self.image_cache_block_manager = factory.get_image_cache_block_manager()
        self.batch_scheduler = factory.get_batch_scheduler()
        self.executor = factory.get_executor(self.kv_cache_block_manager, self.image_cache_block_manager)
        self.request_processor = factory.get_request_processor(self.batch_scheduler)

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