import argparse
from typing import Optional, Literal
from dataclasses import dataclass, field, fields
from hydrainfer.model import ModelFactoryConfig, getModelFactory, ModelFactoryContext
from hydrainfer.memory import TokenCacheBlockManagerConfig, CommunicationBackendManagerConfig
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
    migrate_backend_config: CommunicationBackendManagerConfig = field(default_factory=CommunicationBackendManagerConfig)