import os
import json
import torch
from typing import Any, Literal
from dxz.memory import KVCache
from dataclasses import dataclass, asdict, field
from concurrent.futures import ThreadPoolExecutor
from batch_analysis import BatchDecode, BatchEncode, benchmark
from dxz.model import getModelFactory, ModelFactoryConfig, ModelFactoryContext, VisionModel, VisionModelParameters, LanguageModel, LanguageModelParameters, LanguageModelConfig
from dxz.layer.causal_attention import AttentionParameters, AttentionParametersBuilder

dtype = torch.half
device = torch.device('cuda')
result_path = os.path.join('result', 'concurrency_analsis.json')


@dataclass
class ConcurrenceMetric:
    multi_threads: bool
    multi_streams: bool
    n_encode_batch_size: int
    n_decode_batch_size: int
    n_prefill_batch_size:  int
    latency: float # s

@dataclass
class ConcurrencyAnalysis:
    results: list[ConcurrenceMetric] = field(default_factory=list)

class ConcurrentExecution:
    def __init__(self, multi_threads: bool, multi_streams: bool, vision: BatchEncode, language: BatchDecode) -> None:
        self.pool = thread_pool = ThreadPoolExecutor(max_workers=1)
        self.stream = torch.cuda.Stream()
        self.multi_threads = multi_threads
        self.multi_streams = multi_streams
        self.vision = vision
        self.language = language

    def fowrard_vision(self):
        if self.multi_streams:
            self.vision()
            self.stream.synchronize()
        else:
            self.vision()


    def __call__(self):
        if self.multi_threads:
            future = self.pool.submit(self.fowrard_vision)
            self.language()
            future.result()
            torch.cuda.synchronize()
        else:
            if self.multi_streams:
                with torch.cuda.stream(self.stream):
                    self.vision()
                self.language()
                self.stream.synchronize()
                torch.cuda.synchronize()
            else:
                self.vision()
                self.language()
                torch.cuda.synchronize()
                

if __name__ == '__main__':
    model_name = "llava-hf/llava-1.5-7b-hf"
    model_path = "/mnt/cfs/9n-das-admin/llm_models/llava-1.5-7b-hf"
    model_factory = getModelFactory(ModelFactoryConfig(name=model_name, path=model_path), ModelFactoryContext())
    vision_model = model_factory.getVisionModel()
    language_model = model_factory.getLanguageModel()
    language_config = model_factory.getLanguageModelConfig()

    encode = BatchEncode(vision_model=vision_model, batch_size=8)
    decode = BatchDecode(language_model=language_model, language_config=language_config, batch_size=8, decode_kv_cache_tokens=1024)

    result = ConcurrencyAnalysis()
    print(f'encode latency {benchmark(encode)}')
    print(f'decode latency {benchmark(decode)}')
    for multi_streams in [False, True]:
        for multi_threads in [False, True]:
            op = ConcurrentExecution(
                multi_threads = multi_threads, 
                multi_streams = multi_streams, 
                vision = encode, 
                language = decode, 
            )

            # with nvtx.annotate(f'multi_streams {multi_streams} multi_threads {multi_threads}'):
            latency = benchmark(op)
            
            metric = ConcurrenceMetric(
                multi_threads = multi_threads, 
                multi_streams = multi_streams, 
                n_encode_batch_size = 8, 
                n_decode_batch_size = 8, 
                n_prefill_batch_size = 0, 
                latency = latency, 
            )
            result.results.append(metric)
            print(f'multistream {multi_streams} multithreads {multi_threads} {latency} s')

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as file:
        json.dump(asdict(result), fp=file, indent=4)