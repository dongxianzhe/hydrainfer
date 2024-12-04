from dataclasses import dataclass

@dataclass
class BenchmarkMetrics:
    benchmark_duration: float
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
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
        print("{:<40} {:<10}".format("Total input tokens:", self.total_input))
        print("{:<40} {:<10}".format("Total generated tokens:", self.total_output))
        print("{:<40} {:<10}".format("Mean input tokens:", self.mean_input_len))
        print("{:<40} {:<10}".format("Median input tokens:", self.median_input_len))
        print("{:<40} {:<10}".format("Max input tokens:", self.max_input_len))
        print("{:<40} {:<10}".format("Mean generated tokens:", self.mean_output_len))
        print("{:<40} {:<10}".format("Median generated tokens:", self.median_output_len))
        print("{:<40} {:<10}".format("Max generated tokens:", self.max_output_len))
        print("{s:{c}^{n}}".format(s='Throughput', n=50, c='-'))
        print("{:<40} {:<10.2f}".format("Request throughput (req/s):", self.request_throughput))
        print("{:<40} {:<10.2f}".format("Input token throughput (tok/s):", self.input_throughput))
        print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):", self.output_throughput))
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