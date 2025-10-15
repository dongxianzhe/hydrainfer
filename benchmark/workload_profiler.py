from hydrainfer.utils.torch_utils import get_dtype_size, str2dtype
import torch
import tqdm
import argparse
from dataclasses import dataclass
from synthetic_dataset import SyntheticDataset
import numpy as np
from hydrainfer.model import BatchSizeAnalysisResult, BatchSizeBenchmarkMetric
from hydrainfer.model import BatchSizeProfiler, ModelFactoryConfig, ModelFactoryContext, getModelFactory
from hydrainfer.utils.json_utils import load_json
from hydrainfer.memory import TokenCacheBlockManager, TokenCacheBlockManagerConfig


def partition_nodes(n_nodes: int, ratios: list[float]) -> list[int]:
    num_classes = len(ratios)
    if n_nodes < num_classes:
        raise Exception(f'n_nodes {n_nodes} can not be partitioned with ratios {ratios}')

    ratios = np.array(ratios, dtype=float)
    total_ratio = np.sum(ratios)

    # 1. first allocate each class one node
    base_alloc = np.ones(num_classes, dtype=int)
    remaining = n_nodes - num_classes

    # 2. if ratios is all zero divide zero
    if total_ratio == 0:
        for i in range(remaining):
            base_alloc[i % num_classes] += 1
        return base_alloc.tolist()

    # 3.distribute the remaining part proportionally.
    norm_ratios = ratios / total_ratio
    raw_alloc = norm_ratios * remaining
    int_alloc = np.floor(raw_alloc).astype(int)
    leftover = remaining - np.sum(int_alloc)

    # 4.distribute the remainder, prioritize those with the largest rounding errors.
    residuals = raw_alloc - int_alloc
    for i in np.argsort(-residuals):
        if leftover == 0:
            break
        int_alloc[i] += 1
        leftover -= 1

    final_alloc = base_alloc + int_alloc
    return final_alloc.tolist()


class DisaggregationProfiler:
    def __init__(self, history: SyntheticDataset, n_nodes: int, model_path: str, tpot_slo: float, ttft_slo: float) -> None:
        config, context = ModelFactoryConfig(path=model_path), ModelFactoryContext()
        self.model_factory = getModelFactory(config, context)
        self.tokenizer = self.model_factory.getTokenizer()
        self.image_token_caculator = self.model_factory.getVisionModelConfig().image_token_caculator
        self.n_nodes = n_nodes
        self.history = history
        self.language_config = self.model_factory.getLanguageModelConfig()
        self.model_profiler = self.model_factory.getModelProfiler()

        self.batch_profiler = BatchSizeProfiler(config, context)
        self.ttft_slo = ttft_slo
        self.tpot_slo = tpot_slo
        self.dtype: str = config.dtype

    def estimate_encode_throughput(self, ttft_slo: float) -> BatchSizeBenchmarkMetric:
        l, r = 1, 16
        mid: int = None 
        metric = None
        while l < r:
            mid = (l + r) // 2
            metric = self.batch_profiler.profile_encode(mid)
            if metric.latency < 0.5 * ttft_slo:
                l = mid + 1
            else: 
                r = mid - 1
        print(f'estimate throughput {metric} target latency {0.5 * ttft_slo}')
        return metric

    def estimate_prefill_throughput(self, ttft_slo: float, n_prefill_tokens: int) -> BatchSizeBenchmarkMetric:
        l, r = 1, 8
        mid = None
        metric = None
        while l < r:
            mid = (l + r) // 2
            metric = self.batch_profiler.profile_prefill(mid, n_prompt_tokens=n_prefill_tokens)
            if metric.latency < 0.5 * ttft_slo:
                l = mid + 1
            else:
                r = mid - 1
        print(f'estimate throughput {metric} target latency {0.5 * ttft_slo}')
        return metric

    def estimate_max_running_requests(self, n_kv_cache_tokens: int) -> int:
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        total_memory = props.total_memory
        dtype_size = get_dtype_size(str2dtype(self.dtype))
        each_request_memory = n_kv_cache_tokens * 2 * self.language_config.n_layers * self.language_config.n_kv_heads * self.language_config.head_dim * dtype_size
        model_params = self.model_profiler.profile_model_params()
        max_running_reqeusts: int = int((total_memory - model_params.language_model_parmas * dtype_size) * 0.9 / each_request_memory)
        print(f'estimate_max_running_requests = {max_running_reqeusts}')
        return max_running_reqeusts

    def estimate_decode_throughput(self, tpot_slo: float, n_kv_cache_tokens: str) -> BatchSizeBenchmarkMetric:
        l, r = 1, self.estimate_max_running_requests(n_kv_cache_tokens)
        mid = None
        metric = None
        while l < r:
            mid = (l + r) // 2
            metric = self.batch_profiler.profile_decode(mid, n_kv_cache_tokens)
            if metric.latency < tpot_slo:
                l = mid + 1
            else:
                r = mid - 1
        print(f'estimate throughput {metric} target latency {tpot_slo}')
        return metric

    def profile(self):
        n_requests = len(self.history.entries)
        total_encode_tokens: int = 0
        total_prefill_tokens: int = 0
        total_decode_tokens: int = 0
        for entry in tqdm.tqdm(self.history.entries, desc='caculating worklooad of encode, prefill, decode'):
            n_image_tokens = sum([self.image_token_caculator.get_num_image_tokens(image_size) for image_size in entry.images_size])
            n_text_tokens = len(self.tokenizer.encode(entry.prompt))
            n_input_tokens = n_image_tokens + n_text_tokens
            n_output_tokens = entry.n_output_tokens_ref

            total_encode_tokens += n_image_tokens
            total_prefill_tokens += n_input_tokens
            total_decode_tokens += n_output_tokens

        avg_encode_tokens: int = total_encode_tokens / n_requests
        avg_prefill_tokens: int = total_prefill_tokens / n_requests
        avg_decode_tokens: int = total_decode_tokens / n_requests

        t_e = self.estimate_encode_throughput(ttft_slo=self.ttft_slo).throughput
        t_p = self.estimate_prefill_throughput(ttft_slo=self.ttft_slo, n_prefill_tokens=int(avg_prefill_tokens)).throughput
        t_d = self.estimate_decode_throughput(tpot_slo=self.tpot_slo, n_kv_cache_tokens=int(avg_prefill_tokens + avg_decode_tokens)).throughput

        t1 = total_encode_tokens // t_e
        t2 = total_prefill_tokens // t_p
        t3 = total_decode_tokens // t_d

        partitions = partition_nodes(n_nodes=self.n_nodes, ratios=[t1, t2, t3]) 
        n_e, n_p, n_d = partitions
        print(f'avg_encode_tokens, avg_prefill_tokens, avg_decode_tokens: {avg_encode_tokens} {avg_prefill_tokens} {avg_decode_tokens}')
        print(f'encode_workload, prefill_workload, decode_workload: {t1} {t2} {t3}')
        print(f'e p d instacnes numbers{partitions}')
        print(f'possible deploy params: ')
        print(f"cluster=general cluster.n_enode={n_e} cluster.n_epnode=0 cluster.n_ednode=0 cluster.n_epdnode=0 cluster.n_pnode={n_p} cluster.n_pdnode=0 cluster.n_dnode={n_d}")
        print(f"cluster=general cluster.n_enode=0 cluster.n_epnode={n_e + n_p} cluster.n_ednode=0 cluster.n_epdnode=0 cluster.n_pnode=0 cluster.n_pdnode=0 cluster.n_dnode={n_d}")
        print(f"cluster=general cluster.n_enode=0 cluster.n_epnode=0 cluster.n_ednode={n_e + n_d} cluster.n_epdnode=0 cluster.n_pnode={n_p} cluster.n_pdnode=0 cluster.n_dnode=0")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='profiler workload to get disaggregation methods and instances number', conflict_handler='resolve')
    parser.add_argument(f'--model-path', type=str, default="/models/llava-v1.6-vicuna-7b-hf", help='The path of the model.')
    parser.add_argument(f'--ttft_slo', type=float, default=8)
    parser.add_argument(f'--tpot_slo', type=float, default=0.4)
    parser.add_argument(f'--n_nodes', type=int, default=8)
    SyntheticDataset.add_cli_args(parser)
    args, remain_args = parser.parse_known_args()

    history = SyntheticDataset.from_cli_args(args, num_requests=100)
    profiler = DisaggregationProfiler(history=history, model_path=args.model_path, tpot_slo=args.tpot_slo, ttft_slo=args.ttft_slo, n_nodes=args.n_nodes)
    profiler.profile()