import argparse
from typing import Optional, Literal
from dataclasses import dataclass, field, fields
from dxz.model import ModelFactoryConfig, getModelFactory, ModelFactoryContext
from dxz.memory import TokenCacheBlockManagerConfig
from dxz.engine import RequestProcessorConfig, BatchSchedulerConfig, ExecutorConfig, WorkerConfig, BatchSchedulerProfilerConfig
from dxz.utils.zmq_utils import ZMQConfig

@dataclass
class NCCLMigrateCommunicator:
    host: str
    port: int

@dataclass
class NodeConfig:
    request_processor: RequestProcessorConfig = field(default_factory=RequestProcessorConfig)
    model: ModelFactoryConfig = field(default_factory=ModelFactoryConfig)
    batch_scheduler: BatchSchedulerConfig = field(default_factory=BatchSchedulerConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    batch_scheduler_profiler: BatchSchedulerProfilerConfig = field(default_factory=BatchSchedulerProfilerConfig)
    enable_encode: bool = True
    enable_prefill: bool = True
    enable_decode: bool = True
    zmq: Optional[ZMQConfig] = None
    debug_migrate: bool = True
    nccl_communicator: Optional[NCCLMigrateCommunicator] = None
    intranode_migrate_backend: Literal['ipc', 'nccl'] = 'ipc'
    internode_migrate_backend: Literal['nccl'] = 'nccl'
    log_latency_breakdown: bool = False
    auto_compute_cache_memory: bool = True