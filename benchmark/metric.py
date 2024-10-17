from dataclasses import dataclass

@dataclass
class BenchmarkMetrics:
    completed: int    # completed number of request
    total_input: int  # total input number of tokens
    total_output: int # total output number of tokens
    request_throughput: float # completed / total time
    input_throughput: float   # total_input / total time
    output_throughput: float  # total_output / total time
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