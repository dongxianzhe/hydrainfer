import asyncio
from dxz.model.model_factory import getModelFactory, ModelFactoryConfig, ModelFactoryContext
from dataclasses import dataclass
from dxz.engine.isa import TextFill, MigrateRequest
from dxz.engine.scheduler import BatchScheduler, SchedulerConfig, BatchRequest
from dxz.memory.vanilla_memory_mangement import VinallaMemoryManagementUnit, MemoryConfig, MemoryContext
from dxz.engine.executor import InstructionExecutor, ExecutorConfig, ExecutorContext
from dxz.engine.worker import Worker, getWorker, WorkerConfig, WorkerContext
from dxz.request.rcb import RequestControlBlock
from dxz.cluster.raynode import RayNode

class PNodeConfig:
    pass


class PNodeContext:
    model_factory_config: ModelFactoryConfig
    memory_config: MemoryConfig
    scheduler_config: SchedulerConfig
    executor_config: ExecutorConfig
    worker_config: WorkerConfig


class PNode(RayNode):
    def __init__(self, config: PNodeConfig, context: PNodeContext):
        model_factory = getModelFactory(context.model_factory_config, ModelFactoryContext())
        self.language_model_config = model_factory.getLanguageModelConfig()
        self.scheduler = BatchScheduler(context.scheduler_config)
        self.mmu = VinallaMemoryManagementUnit(context.memory_config, MemoryContext(
            n_layers = self.language_model_config.n_layers, 
            num_kv_heads = self.language_model_config.n_kv_heads, 
            head_size = self.language_model_config.head_dim, 
            dtype = context.model_factory_config.dtype, 
            device = context.model_factory_config.device, 
        ))
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
        batch_fills = BatchRequest()
        batch_migrate = BatchRequest()
        for rcb, inst in batch:
            if isinstance(inst, TextFill):
                batch_fills.append(rcb)
            elif isinstance(inst, MigrateRequest):
                batch_migrate.append(rcb)
            else:
                raise Exception(f'pnode is not supported to execute instruction {inst}')
        self.executor.execute_fill(batch_fills)
        await self.execute_batch_migrate(batch_migrate)

        for rcb, inst in batch:
            if rcb.is_finished():
                for virtual_kv_cache in rcb.virtual_kv_caches:
                    self.mmu.realloc(virtual_kv_cache, 0)
            else:
                self.scheduler.schedule_running([rcb])

    async def step_loop(self):
        while True:
            await self.step()
            await asyncio.sleep(0.001)

    async def register_node(self, node: RayNode):
        self.nodes.append(node)

    async def execute_batch_migrate(self, contexts: BatchRequest):
        if len(contexts) == 0:
            return
        node = self.nodes[0]
        for rcb, _ in contexts:
            rcb.step()
            node.migrate.remote(rcb)