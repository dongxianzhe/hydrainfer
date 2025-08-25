from dataclasses import dataclass
from typing import Literal, Optional
from hydrainfer.cluster import NodeConfig


@dataclass
class InstanceDataParallelConfig:
    node_type: Literal["E", "P", "D", "EP", "ED", "PD", "EPD"]
    node_config: Optional[NodeConfig] = None
    n_replicas: int = 0


@dataclass
class DisaggregationMethod:
    instance_data_parallel_configs: list[InstanceDataParallelConfig]


class DisaggregationMethodProfiler:
    def __init__(self):
        pass

    def profile() -> DisaggregationMethod:
        pass
        # 1. start nodes
        # 2. benchmark
        # 3. analysis result
        # 4. update nodes