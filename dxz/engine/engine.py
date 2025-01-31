import time
from typing import Optional
from dataclasses import dataclass, field, fields
import torch
from dxz.engine.isa import Instruction, Fill, TextFill, ImageFill, Mov, ReAlloc, EmptyInstruction, ImageEmbedFill, ImageEmbed
from dxz.request.rcb import RequestControlBlock
from dxz.memory.memory_management import MemoryManagementUnit, MemoryConfig, MemoryContext, getMemoryManagementUnit
from dxz.engine.scheduler import SchedulerConfig, BatchScheduler
from dxz.model.model_factory import ModelFactory, getModelFactory, ModelFactoryConfig, ModelFactoryContext
from dxz.engine.executor import InstructionExecutor, ExecutorContext, ExecutorConfig
import argparse




@dataclass
class EngineConfig:
    memory_config: MemoryConfig = field(default_factory=MemoryConfig)
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    executor_config: ExecutorConfig = field(default_factory=ExecutorConfig)

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> 'EngineConfig':
        attrs = [attr.name for attr in fields(cls) if attr.name not in ['memory_config', 'scheduler_config', 'executor_config']]
        memory_config = MemoryConfig.from_cli_args(args)
        scheduler_config = SchedulerConfig.from_cli_args(args)
        executor_config = ExecutorConfig.from_cli_args(args)
        config = cls(
            memory_config=memory_config, 
            scheduler_config=scheduler_config, 
            executor_config=executor_config, 
            **{attr: getattr(args, attr) for attr in attrs}
        )
        return config

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = MemoryConfig.add_cli_args(parser)
        parser = SchedulerConfig.add_cli_args(parser)
        parser = ExecutorConfig.add_cli_args(parser)
        return parser


@dataclass
class EngineContext:
    model_factory_config: ModelFactoryConfig


class Engine:
    def __init__(self, config: EngineConfig, context: EngineContext):
        self.config = config
        # memory
        model_factory = getModelFactory(context.model_factory_config, ModelFactoryContext())
        language_model_config = model_factory.getLanguageModelConfig()
        self.memory_context = MemoryContext(
            n_layers = language_model_config.n_layers,
            head_size = language_model_config.head_dim, 
            num_kv_heads = language_model_config.n_kv_heads, 
            dtype = context.model_factory_config.dtype, 
            device = context.model_factory_config.device, 
        )
        self.mmu = getMemoryManagementUnit(
            config = self.config.memory_config, 
            context = self.memory_context, 
        )
        # scheduler
        self.scheduler = BatchScheduler(self.config.scheduler_config)
        # executor
        executor_context = ExecutorContext(
            model_factory_config = context.model_factory_config, 
            block_size = config.memory_config.block_size, 
            mmu = self.mmu
        )
        self.executor = InstructionExecutor(config.executor_config, executor_context)

    def schedule(self, rcbs: list[RequestControlBlock]):
        self.scheduler.schedule_new(rcbs)
    
    @torch.inference_mode()
    def step(self) -> dict[int, int]:
        # 1. schedule requests
        contexts = self.scheduler.step()
        if len(contexts) == 0:
            return {}

        # 2. execute instructions
        fill_contexts = []
        image_embed_contexts = []
        empty_contexts = []
        for context in contexts:
            rcb, instruction = context
            if len(rcb.virtual_kv_caches) == 0:
                rcb.virtual_kv_caches = self.mmu.allocate_virtual_kv_caches(rcb.n_virtual_kv_caches)
            if isinstance(instruction, Fill):
                fill_contexts.append(context)
                continue
            if isinstance(instruction, EmptyInstruction):
                empty_contexts.append(context)
                continue
            if isinstance(instruction, ImageEmbed):
                image_embed_contexts.append(context)
                continue
            raise Exception(f'unsupported instrction {type(instruction)}')

        future = self.executor.execute_image_embed(image_embed_contexts)
        self.executor.execute_fill(fill_contexts)
        if future is not None:
            future.result()
        self.executor.execute_empty(empty_contexts)

        # 3. scheduler requests
        t = time.perf_counter()
        for rcb, _ in contexts:
            if rcb.is_finished():
                rcb.metric.finished_time = t
                for vkvc in rcb.virtual_kv_caches:
                    self.mmu.realloc(vkvc, 0)
            else:
                self.scheduler.schedule_running([rcb])