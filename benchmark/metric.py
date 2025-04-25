import time
import pickle
import numpy as np
from typing import Union
from dataclasses import dataclass, field
from synthetic_dataset import SyntheticDataset, SyntheticDataEntry


@dataclass
class OnlineRequestOutput:
    entry: SyntheticDataEntry
    success: bool = False
    output_text: str = ""
    start_time: float = 0.
    token_times: list[float] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    request_rate: float = 0
    start_time: float = 0
    end_time: float = 0
    outputs: list[OnlineRequestOutput] = field(default_factory=list)


@dataclass
class MethodResults:
    method_name: str
    datasets: dict[str, int]
    model: str
    model_path: str
    results: list[BenchmarkResult] = field(default_factory=list)


@dataclass
class BenchmarkMetrics:
    benchmark_duration: float
    completed: int
    total_output_tokens: int
    request_throughput: float
    output_token_throughput: float

    mean_output_tokens: int
    median_output_tokens: int
    max_output_tokens: int

    mean_latency_ms: float
    median_latency_ms: float
    p90_latency_ms: float
    p99_latency_ms: float

    mean_ttft_ms: float
    median_ttft_ms: float
    p90_ttft_ms: float
    p99_ttft_ms: float

    mean_tpot_ms: float
    median_tpot_ms: float
    p90_tpot_ms: float
    p99_tpot_ms: float

    ttft_slo_attainment: float
    tpot_slo_attainment: float
    slo_attainment: float

    
class BenchmarkMetricsBuilder:
    def __init__(self, 
        start_time: float, 
        end_time  : float, 
    ):
        self.start_time = start_time
        self.end_time = end_time
        self.total_requests: int = 0
        self.completed = 0
        self.num_output_tokens: list[int] = []
        self.latencies: list[float] = []
        self.ttfts: list[float] = []
        self.tpots: list[float] = []
        self.ttft_slo = 0.
        self.tpot_slo = 0.
        self.ttft_slo_cnt: int = 0 # num of requests that satisfied ttft
        self.tpot_slo_cnt: int = 0 # num of requests that satisfied tpot
        self.slo_cnt: int = 0 # num of requests that satisfied both ttft and tpot
        
    def set_ttft_slo(self, ttft_slo: float):
        self.ttft_slo = ttft_slo

    def set_tpot_slo(self, tpot_slo: float):
        self.tpot_slo = tpot_slo

    def _append(
        self, 
        success: bool, # wheather request is success
        arrival_time: float,
        finished_time: float,
        token_times: list[float] # request each token finish time
    ):
        assert self.ttft_slo > 0., 'set ttft_slo first and append request'
        assert self.tpot_slo > 0., 'set tpot_slo first and append request'
        self.total_requests += 1
        if not success:
            return

        self.completed += 1
        self.num_output_tokens.append(len(token_times))
        self.latencies.append(finished_time - arrival_time)

        is_ttft_satisfied: bool = True
        is_tpot_satisfied: bool = True
        tpot_above_slo_cnt: int = 0
        for i in range(len(token_times)):
            if i == 0:
                ttft = token_times[i] - arrival_time
                self.ttfts.append(ttft) 
                is_ttft_satisfied = ttft < self.ttft_slo
            else:
                tpot = token_times[i] - token_times[i - 1]
                self.tpots.append(tpot)
                tpot_above_slo_cnt += tpot > self.tpot_slo
                is_tpot_satisfied = is_tpot_satisfied and tpot_above_slo_cnt <= 0
        self.ttft_slo_cnt += is_ttft_satisfied
        self.tpot_slo_cnt += is_tpot_satisfied
        self.slo_cnt += is_ttft_satisfied and is_tpot_satisfied 

    def append(self, data: Union[OnlineRequestOutput, list[OnlineRequestOutput]]):
        if isinstance(data, list):
            for output in data:
                self.append(output)
            return
        if isinstance(data, OnlineRequestOutput):
            self._append(
                success = data.success, 
                arrival_time = data.start_time, 
                finished_time = data.token_times[-1], 
                token_times = data.token_times, 
            )
            return
        raise Exception(f'invalid data dtype {type(data)}')

    def get_metrics(self) ->  BenchmarkMetrics:
        duration = self.end_time - self.start_time
        metrics = BenchmarkMetrics(
            benchmark_duration=duration, 
            completed=self.completed,
            total_output_tokens=sum(self.num_output_tokens),
            mean_output_tokens=np.mean(self.num_output_tokens),
            median_output_tokens=np.median(self.num_output_tokens),
            max_output_tokens=max(self.num_output_tokens),
            request_throughput=self.completed / duration,
            input_token_throughput=sum(self.input_lens) / duration,
            output_token_throughput=sum(self.output_lens) / duration,
            mean_latency_ms=np.mean(self.latencies) * 1000,
            median_latency_ms=np.median(self.latencies) * 1000,
            p90_latency_ms=np.percentile(self.latencies, 90) * 1000,
            p99_latency_ms=np.percentile(self.latencies, 99) * 1000,
            mean_ttft_ms=np.mean(self.ttfts or 0) * 1000,
            median_ttft_ms=np.median(self.ttfts or 0) * 1000,
            p90_ttft_ms=np.percentile(self.ttfts or 0, 90) * 1000,
            p99_ttft_ms=np.percentile(self.ttfts or 0, 99) * 1000,
            mean_tpot_ms=np.mean(self.tpots) * 1000,
            median_tpot_ms=np.median(self.tpots) * 1000,
            p90_tpot_ms=np.percentile(self.tpots, 90) * 1000 if len(self.tpots) > 0 else np.nan,
            p99_tpot_ms=np.percentile(self.tpots, 99) * 1000 if len(self.tpots) > 0 else np.nan,
            ttft_slo_attainment = self.ttft_slo_cnt / self.total_requests,
            tpot_slo_attainment = self.tpot_slo_cnt / self.total_requests,
            slo_attainment = self.slo_cnt / self.total_requests, 
        )
        return metrics