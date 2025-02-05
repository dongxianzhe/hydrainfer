import asyncio
import torch
from dataclasses import dataclass
from dxz.utils.zmq_utils import init_zmq_send
from dxz.model.model_factory import getModelFactory, ModelFactoryConfig, ModelFactoryContext
from dxz.memory.vanilla_memory_mangement import VinallaMemoryManagementUnit
from dxz.memory.memory_management import MemoryConfig, MemoryContext
from dxz.engine.scheduler import BatchScheduler, SchedulerConfig, BatchRequest
from dxz.engine.isa import TextFill, MigrateRequest, EmptyInstruction
from dxz.engine.executor import InstructionExecutor, ExecutorContext, ExecutorConfig
from dxz.engine.worker import Worker, WorkerConfig, getWorker,WorkerContext
from dxz.request.rcb import RequestControlBlock, OutputTokenProcessor, LogOutputTokenProcessor
from dxz.request.offline_inference_output import OfflineInferenceOutput
from dxz.cluster.raynode import RayNode

@dataclass
class DNodeConfig:
    pass


@dataclass
class DNodeContext:
    zmq_url: str
    model_factory_config: ModelFactoryConfig
    memory_config: MemoryConfig
    scheduler_config: SchedulerConfig
    executor_config: ExecutorConfig
    worker_config: WorkerConfig


class DNode(RayNode):
    def __init__(self, config: DNodeConfig, context: DNodeContext):
        self.zmq_send = init_zmq_send(zmq_url=context.zmq_url)
        model_factory = getModelFactory(context.model_factory_config, ModelFactoryContext())
        self.tokenizer = model_factory.getTokenizer()
        self.language_model_config = model_factory.getLanguageModelConfig()
        self.mmu = VinallaMemoryManagementUnit(
            context.memory_config, MemoryContext(
                n_layers = self.language_model_config.n_layers, 
                num_kv_heads = self.language_model_config.n_kv_heads, 
                head_size = self.language_model_config.head_dim, 
                dtype = context.model_factory_config.dtype, 
                device = context.model_factory_config.device, 
            )
        )
        self.scheduler = BatchScheduler(context.scheduler_config)

        self.worker = getWorker(context.worker_config, WorkerContext(
            model_factory_config=context.model_factory_config
        ))
        self.executor = InstructionExecutor(context.executor_config, ExecutorContext(
            model_factory_config = context.model_factory_config, 
            block_size = context.memory_config.block_size, 
            mmu = self.mmu, 
            worker=self.worker, 
        ))
        self.nodes: list[RayNode] = []

    async def step(self):
        batch: BatchRequest = self.scheduler.step()
        if len(batch) == 0:
            return
        batch_fill = BatchRequest()
        batch_empty = BatchRequest()
        for rcb, inst in batch:
            if isinstance(inst, TextFill):
                batch_fill.append(rcb)
            elif isinstance(inst, EmptyInstruction):
                batch_empty.append(rcb)
            else:
                raise Exception(f'{inst} is not supported in dnode')
        self.executor.execute_fill(batch_fill)
        self.executor.execute_empty(batch_empty)
        for rcb, inst in batch:
            if rcb.is_finished():
                for virtual_kv_cache in rcb.virtual_kv_caches:
                    self.mmu.realloc(virtual_kv_cache, 0)
                for output_token_processor in rcb.output_token_processors:
                    if isinstance(output_token_processor, LogOutputTokenProcessor):
                        text = self.tokenizer.decode(output_token_processor.token_ids)
                        self.zmq_send.send_pyobj(OfflineInferenceOutput(text=text))
            else:
                self.scheduler.schedule_running([rcb])

    async def step_loop(self):
        while True:
            await self.step()
            await asyncio.sleep(0.001)

    async def register_node(self, node: "RayNode"): 
        self.nodes.append(node)

    async def migrate(self,rcb: RequestControlBlock):
        print(rcb.instructions)
        # 1. allocate kv cache for request
        dst_virtual_kv_caches = self.mmu.allocate_virtual_kv_caches(rcb.n_virtual_kv_caches)
        for src_virtual_kv_cache, dst_virtual_kv_cache in zip(rcb.virtual_kv_caches, dst_virtual_kv_caches):
            # 2. alloate blocks for each layer
            self.mmu.realloc(dst_virtual_kv_cache, src_virtual_kv_cache.n_kv_cache_tokens)
            # 3. copy kv cache data from send node
            self.mmu.migrate_blocks(src_virtual_kv_cache, dst_virtual_kv_cache)
        # 4. schedule
        rcb.virtual_kv_caches = dst_virtual_kv_caches
        self.scheduler.schedule_new([rcb])