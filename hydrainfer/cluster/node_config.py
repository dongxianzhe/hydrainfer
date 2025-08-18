import argparse
from typing import Optional, Literal
from dataclasses import dataclass, field, fields
from hydrainfer.model import ModelFactoryConfig, getModelFactory, ModelFactoryContext
from hydrainfer.memory import TokenCacheBlockManagerConfig
from hydrainfer.engine import RequestProcessorConfig, BatchSchedulerConfig, ExecutorConfig, WorkerConfig, BatchSchedulerProfilerConfig
from hydrainfer.utils.zmq_utils import ZMQConfig

@dataclass
class NCCLCommunicatorConfig:
    host: str
    port: int

@dataclass
class NodeConfig:
    request_processor: RequestProcessorConfig = field(default_factory=RequestProcessorConfig)
    model: ModelFactoryConfig = field(default_factory=ModelFactoryConfig)
    batch_scheduler: BatchSchedulerConfig = field(default_factory=BatchSchedulerConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    batch_scheduler_profiler: BatchSchedulerProfilerConfig = field(default_factory=BatchSchedulerProfilerConfig)
    zmq: Optional[ZMQConfig] = None
    debug_migrate: bool = True
    nccl_communicator: Optional[NCCLCommunicatorConfig] = None
    intranode_migrate_backend: Literal['ipc', 'nccl'] = 'ipc'
    internode_migrate_backend: Literal['nccl'] = 'nccl'
    log_latency_breakdown: bool = False