from typing import Optional
from dataclasses import dataclass, field
from synthetic_dataset import SyntheticDataEntry


@dataclass
class Statistics:
    max: Optional[float] = None
    min: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    p90: Optional[float] = None
    p99: Optional[float] = None
    var: Optional[float] = None


@dataclass
class OnlineRequestOutput:
    entry: SyntheticDataEntry
    success: bool = False
    error_msg: str = ""
    output_text: str = ""
    start_time: float = 0.
    token_times: list[float] = field(default_factory=list)
    # The following data is derived from the data above. 
    total_tokens: int = 0
    latency: Optional[float] = None
    ttft: Optional[float] = None
    tpots: list[float] = field(default_factory=list)
    tpot_statistics: Optional[Statistics] = None


@dataclass
class BenchmarkResult:
    request_rate: float = 0
    start_time: float = 0
    end_time: float = 0
    outputs: list[OnlineRequestOutput] = field(default_factory=list)
    # The following data is derived from the data above.
    total_tokens: int = 0
    total_success: int = 0
    token_throughput: float = 0
    request_throughput: float = 0
    latencies: list[float] = field(default_factory=list)
    ttfts: list[float] = field(default_factory=list)
    tpots: list[float] = field(default_factory=list)
    latency_statistics: Optional[Statistics] = None
    ttft_statistics: Optional[Statistics] = None
    tpot_statistics: Optional[Statistics] = None


@dataclass
class MethodResults:
    method_name: Optional[str]
    datasets: dict[str, int]
    model: str
    model_path: str
    results: list[BenchmarkResult] = field(default_factory=list)

