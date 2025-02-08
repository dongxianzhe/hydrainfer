import time
import torch
import argparse
from dataclasses import dataclass, field, fields
from dxz.request import Request
from dxz.model import getModelFactory, ModelFactoryConfig, ModelFactoryContext
from dxz.engine import RequestProcessor, RequestProcessorConfig, RequestProcessorContext, ImageEmbed, Fill, EmptyInstruction, BatchScheduler, BatchSchedulerConfig, ExecutorConfig, WorkerConfig, getWorker, WorkerContext, ExecutorContext, InstructionExecutor, Engine, BatchRequest, RequestProcessParameters, EngineComponentFactory
from dxz.memory import TokenCacheBlockManager, TokenCacheBlockManagerContext

@dataclass
class EPDNodeConfig:
    request_processor_config: RequestProcessorConfig = field(default_factory=RequestProcessorConfig)
    model_factory_config: ModelFactoryConfig = field(default_factory=ModelFactoryConfig)
    batch_scheduler_config: BatchSchedulerConfig = field(default_factory=BatchSchedulerConfig)
    executor_config: ExecutorConfig = field(default_factory=ExecutorConfig)
    worker_config: WorkerConfig = field(default_factory=WorkerConfig)
    n_kv_blocks: int = 512
    n_image_blocks: int = 16

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> 'EPDNodeConfig':
        attrs = [attr.name for attr in fields(cls) if attr.name not in ['request_processor_config', 'model_factory_config', 'batch_scheduler_config', 'executor_config', 'worker_config',]]
        model_factory_config = ModelFactoryConfig.from_cli_args(args)
        request_processor_config = RequestProcessorConfig.from_cli_args(args)
        batch_scheduler_config = BatchSchedulerConfig.from_cli_args(args)
        executor_config = ExecutorConfig.from_cli_args(args)
        worker_config = WorkerConfig.from_cli_args(args)
        config = cls(
            request_processor_config = request_processor_config, 
            model_factory_config     = model_factory_config, 
            batch_scheduler_config   = batch_scheduler_config, 
            executor_config          = executor_config, 
            worker_config            = worker_config, 
            **{attr: getattr(args, attr) for attr in attrs}
        )
        return config

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = ModelFactoryConfig.add_cli_args(parser)
        parser = RequestProcessorConfig.add_cli_args(parser)
        parser = BatchSchedulerConfig.from_cli_args(parser)
        parser = ExecutorConfig.from_cli_args(parser)
        parser.add_argument('--n-kv-blocks', type=int, default=512, help='number kv cache blocks')
        parser.add_argument('--n-image-blocks', type=int, default=16, help='number image cache blocks')
        parser = WorkerConfig.from_cli_args(parser)
        return parser


@dataclass
class EPDNodeContext:
    pass


class EPDNode(Engine):
    def __init__(self, config: EPDNodeConfig, context: EPDNodeContext):
        self.config = config
        factory = EngineComponentFactory(
            request_processor_config = config.request_processor_config, 
            model_factory_config     = config.model_factory_config, 
            batch_scheduler_config   = config.batch_scheduler_config, 
            executor_config          = config.executor_config, 
            worker_config            = config.worker_config, 
            n_kv_blocks              = config.n_kv_blocks, 
            n_image_blocks           = config.n_image_blocks
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