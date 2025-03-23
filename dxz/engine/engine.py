from typing import Optional
from dxz.request import Request
from dxz.engine import RequestControlBlock, RequestProcessParameters, RequestProcessorConfig, BatchSchedulerConfig, ExecutorConfig, WorkerConfig, BatchScheduler, getWorker, WorkerContext, InstructionExecutor, ExecutorContext, RequestProcessor, Worker
from dxz.model import ModelFactoryConfig, getModelFactory, ModelFactoryContext
from dxz.memory import TokenCacheBlockManager, TokenCacheBlockManagerContext, TokenCacheBlockManagerConfig


class Engine:
    def add_request(self, request: Request, params: RequestProcessParameters):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError


class AsyncEngine:
    async def add_request(self, request: Request, params: RequestProcessParameters):
        raise NotImplementedError

    async def step(self):
        raise NotImplementedError

    async def step_loop(self):
        raise NotImplementedError

    async def register_node(self, node: "AsyncEngine"): 
        raise NotImplementedError

    async def migrate(self, rcb: RequestControlBlock):
        raise NotImplementedError