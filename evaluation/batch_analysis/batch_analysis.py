import os
import json
import time
import torch
from dataclasses import dataclass, field, asdict
from hydrainfer.memory import KVCache
from hydrainfer.model import getModelFactory, ModelFactoryConfig, ModelFactoryContext, VisionModel, VisionModelParameters, LanguageModel, LanguageModelParameters, LanguageModelConfig
from hydrainfer.layer.causal_attention import AttentionParameters, AttentionParametersBuilder


dtype = torch.half
device = torch.device('cuda')


result_path = os.path.join('result', 'batchsize_analsis.json')


@dataclass
class BatchSizeBenchmarkMetric:
    name: str
    batch_size: int
    latency: float  # s
    throughput: int # token /s 


@dataclass
class BatchSizeAnalysis:
    encode: list[BatchSizeBenchmarkMetric] = field(default_factory=list)
    prefill: list[BatchSizeBenchmarkMetric] = field(default_factory=list)
    decode: list[BatchSizeBenchmarkMetric] = field(default_factory=list)


@torch.inference_mode()
def benchmark(forward_fn, warmup=3, n_iter=100):
    for _ in range(warmup):
        forward_fn()
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(n_iter):
        forward_fn()
        torch.cuda.synchronize()
    end = time.time()

    latency = (end - start) / n_iter
    return latency


class BatchEncode:
    def __init__(self, vision_model: VisionModel, batch_size: int) -> None:
        self.batch_size = batch_size
        self.vision_model = vision_model
        self.pixel_values = [torch.randn(size=(1, 3, 336, 336), dtype=dtype, device=device) for i in range(batch_size)]
        self.vision_model_parameters = VisionModelParameters(original_image_sizes=[(336, 336) for i in range(batch_size)])
    def __call__(self):
        self.vision_model.forward(self.pixel_values, self.vision_model_parameters)

class BatchPrefill:
    def __init__(self, language_model: LanguageModel, language_config: LanguageModelConfig, batch_size: int, prefill_prompt_tokens: int):
        self.batch_size = batch_size
        self.language_model = language_model
        self.language_config = language_config
        n_tokens = batch_size * prefill_prompt_tokens
        self.input_ids = torch.randint(1, 10, size=(n_tokens, ), dtype=torch.int, device=device)
        self.image_features = None
        self.position_ids = torch.arange(prefill_prompt_tokens, dtype=torch.int, device=device).repeat(batch_size).reshape(-1)
        selected_token_ids = list(range(prefill_prompt_tokens - 1, batch_size * prefill_prompt_tokens, prefill_prompt_tokens))
        n_layers = self.language_config.n_layers
        n_tokens = 2
        block_size = 16
        n_blocks = (batch_size * prefill_prompt_tokens + block_size - 1) // block_size
        n_kv_heads = self.language_config.n_kv_heads
        head_size = self.language_config.head_dim
        builder = AttentionParametersBuilder(
            num_qo_heads=language_config.n_qo_heads, 
            num_kv_heads=language_config.n_kv_heads, 
            head_dim=language_config.head_dim, 
            block_size=block_size, 
            device=device, 
        )
        self.kv_cache = torch.randn(size=(n_layers, n_tokens, n_blocks, block_size, n_kv_heads, head_size), dtype=dtype, device=device)
        for i in range(self.language_config.n_layers):
            layer_cache = KVCache(
                key_cache=self.kv_cache[i, 0, :, :, :, :],
                value_cache=self.kv_cache[i, 1, :, :, :, :]
            )
            builder.add_kv_cache(layer_cache)
        
        for i in range(batch_size):
            new_cache_slots = torch.tensor(list(range(i * prefill_prompt_tokens, (i + 1) * prefill_prompt_tokens)), dtype=torch.int, device=device)
            block_table = torch.tensor(list(range(n_blocks)), dtype=torch.int, device=device)
            builder.add_request(
                q_seq_len = prefill_prompt_tokens, 
                kv_seq_len = 0, 
                new_cache_slots = new_cache_slots, 
                block_table = block_table, 
            )
        self.model_params = LanguageModelParameters(
            attention_params = builder.build_attention_parameters(), 
            all_sequences_decode = False, 
            selected_token_ids = selected_token_ids
        )

    def __call__(self):
        self.language_model.forward(input_ids=self.input_ids, image_features=self.image_features, position_ids=self.position_ids, model_params=self.model_params)


class BatchDecode:
    def __init__(self, language_model: LanguageModel, language_config: LanguageModelConfig, batch_size: int, decode_kv_cache_tokens: int):
        self.batch_size = batch_size
        self.language_model = language_model
        self.language_config = language_config
        self.input_ids = torch.ones(size=(batch_size, ), dtype=torch.int, device=device)
        self.language_model
        self.image_features = None
        self.position_ids = tensor = torch.full((batch_size,), decode_kv_cache_tokens, dtype=torch.int, device=device)
        selected_token_ids = torch.tensor(list(range(batch_size)), dtype=torch.int, device=device)
        n_layers = self.language_config.n_layers
        n_tokens = 2
        block_size = 16
        # to analsis large batchsize, we share same kv cache for all requests
        n_blocks = (batch_size + decode_kv_cache_tokens + block_size - 1) // block_size
        n_kv_heads = self.language_config.n_kv_heads
        head_size = self.language_config.head_dim
        builder = AttentionParametersBuilder(
            num_qo_heads=language_config.n_qo_heads, 
            num_kv_heads=language_config.n_kv_heads, 
            head_dim=language_config.head_dim, 
            block_size=block_size, 
            device=device, 
        )
        self.kv_cache = torch.randn(size=(n_layers, n_tokens, n_blocks, block_size, n_kv_heads, head_size), dtype=dtype, device=device)
        for i in range(self.language_config.n_layers):
            layer_cache = KVCache(
                key_cache=self.kv_cache[i, 0, :, :, :, :],
                value_cache=self.kv_cache[i, 1, :, :, :, :]
            )
            builder.add_kv_cache(layer_cache)
        
        for i in range(batch_size):
            block_table = list(range(n_blocks))
            builder.add_request(
                q_seq_len = 1, 
                kv_seq_len = 0, 
                new_cache_slots = [i], 
                block_table = block_table, 
            )
        self.model_params = LanguageModelParameters(
            attention_params = builder.build_attention_parameters(), 
            all_sequences_decode = False, 
            selected_token_ids = selected_token_ids
        )

    def __call__(self):
        self.language_model.forward(self.input_ids, self.image_features, self.position_ids, self.model_params)


if __name__ == '__main__':
    max_encode_batch_size = 8
    max_prefill_batch_size = 8
    decode_batch_size_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    prefill_prompt_tokens = 1024
    decode_kv_cache_tokens = 1024
    model_name = "llava-hf/llava-1.5-7b-hf"
    model_path = "/mnt/cfs/9n-das-admin/llm_models/llava-1.5-7b-hf"
    model_factory = getModelFactory(ModelFactoryConfig(name=model_name, path=model_path), ModelFactoryContext())
    vision_model = model_factory.getVisionModel()
    language_model = model_factory.getLanguageModel()
    language_config = model_factory.getLanguageModelConfig()

    results = BatchSizeAnalysis()
    for batch_size in range(1, max_encode_batch_size + 1):
        op = BatchEncode(vision_model, batch_size)
        latency = benchmark(op)
        throughput = batch_size * 576 / latency
        metric = BatchSizeBenchmarkMetric(
            name = 'encode', 
            batch_size = batch_size, 
            latency = latency, 
            throughput = throughput, 
        )
        print(f'encode {batch_size} {latency} s {throughput} token / s')
        results.encode.append(metric)

    for batch_size in range(1, max_prefill_batch_size + 1):
        op = BatchPrefill(language_model, language_config, batch_size, prefill_prompt_tokens)
        latency = benchmark(op)
        throughput = batch_size * prefill_prompt_tokens / latency
        metric = BatchSizeBenchmarkMetric(
            name = 'prefill', 
            batch_size = batch_size, 
            latency = latency, 
            throughput = throughput, 
        )
        print(f'prefill {batch_size} {latency} s {throughput} token / s')
        results.prefill.append(metric)

    for batch_size in decode_batch_size_list:
        op = BatchDecode(language_model, language_config, batch_size, decode_kv_cache_tokens)
        latency = benchmark(op)
        throughput = batch_size / latency
        metric = BatchSizeBenchmarkMetric(
            name = 'decode', 
            batch_size = batch_size, 
            latency = latency, 
            throughput = throughput, 
        )
        results.decode.append(metric)
        print(f'decode {batch_size} {latency} s {throughput} token / s')

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as file:
        json.dump(asdict(results), fp=file, indent=4)