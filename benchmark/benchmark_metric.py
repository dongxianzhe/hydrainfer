import time
import numpy as np
from dataclasses import dataclass, field


@dataclass
class BenchmarkMetrics:
    benchmark_duration: float
    completed: int
    total_input_tokens: int
    total_output_tokens: int
    request_throughput: float
    input_token_throughput: float
    output_token_throughput: float

    mean_input_len: int
    median_input_len: int
    max_input_len: int

    mean_output_len: int
    median_output_len: int
    max_output_len: int

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

    def print(self):
        print("{s:{c}^{n}}".format(s=' Serving Benchmark Result ', n=50, c='='))
        print("{:<40} {:<10}".format("Successful requests:", self.completed))
        print("{:<40} {:<10.2f}".format("Benchmark duration (s):", self.benchmark_duration))
        print("{:<40} {:<10}".format("Total input tokens:", self.total_input_tokens))
        print("{:<40} {:<10}".format("Total generated tokens:", self.total_output_tokens))
        print("{:<40} {:<10}".format("Mean input tokens:", self.mean_input_len))
        print("{:<40} {:<10}".format("Median input tokens:", self.median_input_len))
        print("{:<40} {:<10}".format("Max input tokens:", self.max_input_len))
        print("{:<40} {:<10}".format("Mean generated tokens:", self.mean_output_len))
        print("{:<40} {:<10}".format("Median generated tokens:", self.median_output_len))
        print("{:<40} {:<10}".format("Max generated tokens:", self.max_output_len))
        print("{s:{c}^{n}}".format(s='Throughput', n=50, c='-'))
        print("{:<40} {:<10.2f}".format("Request throughput (req/s):", self.request_throughput))
        print("{:<40} {:<10.2f}".format("Input token throughput (tok/s):", self.input_token_throughput))
        print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):", self.output_token_throughput))
        print("{s:{c}^{n}}".format(s='Time to Latency', n=50, c='-'))
        print("{:<40} {:<10.2f}".format("Mean Latency (ms):", self.mean_latency_ms))
        print("{:<40} {:<10.2f}".format("Median Latency (ms):", self.median_latency_ms))
        print("{:<40} {:<10.2f}".format("P90 Latency (ms):", self.p90_latency_ms))
        print("{:<40} {:<10.2f}".format("P99 Latency (ms):", self.p99_latency_ms))
        print("{s:{c}^{n}}".format(s='Time to First Token', n=50, c='-'))
        print("{:<40} {:<10.2f}".format("Mean TTFT (ms):", self.mean_ttft_ms))
        print("{:<40} {:<10.2f}".format("Median TTFT (ms):", self.median_ttft_ms))
        print("{:<40} {:<10.2f}".format("P90 TTFT (ms):", self.p90_ttft_ms))
        print("{:<40} {:<10.2f}".format("P99 TTFT (ms):", self.p99_ttft_ms))
        print("{s:{c}^{n}}".format(s='Time per Output Token (excl. 1st token)', n=50, c='-'))
        print("{:<40} {:<10.2f}".format("Mean TPOT (ms):", self.mean_tpot_ms))
        print("{:<40} {:<10.2f}".format("Median TPOT (ms):",
                                        self.median_tpot_ms))
        print("{:<40} {:<10.2f}".format("P90 TPOT (ms):", self.p90_tpot_ms))
        print("{:<40} {:<10.2f}".format("P99 TPOT (ms):", self.p99_tpot_ms))
        print("=" * 50)
    
class BenchmarkMetricsBuilder:
    def __init__(self):
        self.completed = 0
        self.input_lens: list[int] = []
        self.output_lens: list[int] = []
        self.latencies: list[float] = []
        self.ttfts: list[float] = []
        self.tpots: list[float] = []
        self.start_time = time.perf_counter()
    
    def append(self, input_len: int, success: bool, output_len: int, arrival_time: float, finished_time: float, token_times: list[float]):
        if success:
            self.completed += 1
            self.input_lens.append(input_len)
            self.output_lens.append(output_len)
            self.latencies.append(finished_time - arrival_time)
            for i in range(len(token_times)):
                if i == 0:
                    self.ttfts.append(token_times[i] - arrival_time) 
                else:
                    self.tpots.append(token_times[i] - token_times[i - 1])
        else:
            self.input_lens.append(input_len)
        self.end_time = time.perf_counter()

    def get_metrics(self) ->  BenchmarkMetrics:
        duration = self.end_time - self.start_time
        metrics = BenchmarkMetrics(
            benchmark_duration=duration, 
            completed=self.completed,
            total_input_tokens=sum(self.input_lens),
            total_output_tokens=sum(self.output_lens),
            mean_input_len=np.mean(self.input_lens),
            median_input_len=np.median(self.input_lens),
            max_input_len=max(self.input_lens),
            mean_output_len=np.mean(self.output_lens),
            median_output_len=np.median(self.output_lens),
            max_output_len=max(self.output_lens),
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
        )
        return metrics