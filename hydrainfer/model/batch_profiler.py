import os
import json
import torch
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from dataclasses import dataclass, field, asdict
from hydrainfer.utils.image_utils import make_random_image
from hydrainfer.memory import KVCache
from hydrainfer.model import getModelFactory, ModelFactory, ModelFactoryConfig, ModelFactoryContext, VisionModel, VisionModelParameters, LanguageModel, LanguageModelParameters, LanguageModelConfig
from hydrainfer.layer.causal_attention import AttentionParameters, AttentionParametersBuilder
from hydrainfer.utils.benchmark_utils import benchmark
from hydrainfer.utils.logger import getLogger
logger = getLogger(__file__)


@dataclass
class BatchSizeBenchmarkMetric:
    name: str
    batch_size: int
    latency: float  # s
    throughput: float # token /s 


@dataclass
class BatchSizeAnalysisResult:
    encode: list[BatchSizeBenchmarkMetric] = field(default_factory=list)
    prefill: list[BatchSizeBenchmarkMetric] = field(default_factory=list)
    decode: list[BatchSizeBenchmarkMetric] = field(default_factory=list)


class BatchSizeProfiler:
    def __init__(self, model_config: ModelFactoryConfig, model_context: ModelFactoryContext):
        self.model_config = model_config
        self.model_context = model_context
        self.model_factory = getModelFactory(model_config, model_context)
        from hydrainfer.utils.torch_utils import str2device, str2dtype
        self.dtype = str2dtype(self.model_config.dtype)
        self.device = str2device(self.model_config.device)

        self.processor = self.model_factory.getProcessor()
        self.image_token_caculator = self.model_factory.getVisionModelConfig().image_token_caculator
        self.vision_model = self.model_factory.getVisionModel()

        self.language_model = self.model_factory.getLanguageModel()
        self.language_config = self.model_factory.getLanguageModelConfig()

    def profile_encode(self, batch_size: int) -> BatchSizeBenchmarkMetric:
        images_size = [(336, 336) for i in range(batch_size)]
        images = [make_random_image(height=height, width=width, n_channel=3) for height, width in images_size]
        pixel_values = [self.processor.process(image).to(self.dtype).to(self.device) for image in images]
        vision_model_parameters = VisionModelParameters(original_image_sizes=images_size)
        
        def forward():
            self.vision_model.forward(pixel_values, vision_model_parameters)

        latency = benchmark(forward, n_iter=20)
        throughput = sum([self.image_token_caculator.get_num_image_tokens((height, width)) for height, width in images_size]) / latency
        metric = BatchSizeBenchmarkMetric(
            name='encode', 
            batch_size=batch_size, 
            latency=latency, 
            throughput=throughput, 
        )
        logger.info(f'batch profile result {metric}')
        return metric

    def profile_encode_list(self, batch_sizes: list[int]) -> list[BatchSizeBenchmarkMetric]:
        return [self.profile_encode(batch_size) for batch_size in  batch_sizes]

    def profile_prefill(self, batch_size: int, n_prompt_tokens: int) -> BatchSizeBenchmarkMetric:
        input_ids = torch.randint(1, 10, size=(batch_size * n_prompt_tokens, ), dtype=torch.int, device=self.device)
        image_features = None
        position_ids = torch.arange(n_prompt_tokens, dtype=torch.int, device=self.device).repeat(batch_size).reshape(-1)
        selected_token_ids = list(range(n_prompt_tokens - 1, batch_size * n_prompt_tokens, n_prompt_tokens))
        n_layers = self.language_config.n_layers
        n_tokens = 2
        block_size = 16
        n_blocks = (batch_size * n_prompt_tokens + block_size - 1) // block_size
        n_kv_heads = self.language_config.n_kv_heads
        head_size = self.language_config.head_dim
        builder = AttentionParametersBuilder(
            num_qo_heads=self.language_config.n_qo_heads, 
            num_kv_heads=self.language_config.n_kv_heads, 
            head_dim=self.language_config.head_dim, 
            block_size=block_size, 
            device=self.device, 
        )
        kv_cache = torch.randn(size=(n_layers, n_tokens, n_blocks, block_size, n_kv_heads, head_size), dtype=self.dtype, device=self.device)
        for i in range(self.language_config.n_layers):
            layer_cache = KVCache(
                key_cache=kv_cache[i, 0, :, :, :, :],
                value_cache=kv_cache[i, 1, :, :, :, :]
            )
            builder.add_kv_cache(layer_cache)
        
        for i in range(batch_size):
            new_cache_slots = torch.tensor(list(range(i * n_prompt_tokens, (i + 1) * n_prompt_tokens)), dtype=torch.int, device=self.device)
            block_table = torch.tensor(list(range(n_blocks)), dtype=torch.int, device=self.device)
            builder.add_request(
                q_seq_len = n_prompt_tokens, 
                kv_seq_len = 0, 
                new_cache_slots = new_cache_slots, 
                block_table = block_table, 
            )
        model_params = LanguageModelParameters(
            input_ids_or_input_embeds=input_ids, 
            position_ids=position_ids, 
            image_features=None, 
            image_overwrite_mask=None, 
            attention_params=builder.build_attention_parameters(), 
            all_sequences_decode=False, 
            selected_token_ids=selected_token_ids
        )
        def forward():
            self.language_model.forward(input_ids=input_ids, image_features=image_features, position_ids=position_ids, model_params=model_params)

        latency = benchmark(forward, n_iter=20)
        throughput = batch_size * n_prompt_tokens / latency
        metric = BatchSizeBenchmarkMetric(
            name = 'prefill', 
            batch_size = batch_size, 
            latency = latency, 
            throughput = throughput, 
        )
        logger.info(f'batch profile result {metric}')
        return metric

    def profile_prefill_list(self, batch_sizes: list[int], n_prompt_tokens: int) -> list[BatchSizeBenchmarkMetric]:
        return [self.profile_prefill(batch_size, n_prompt_tokens) for batch_size in  batch_sizes]

    def profile_decode(self, batch_size: int, n_kv_cache_tokens: int) -> BatchSizeBenchmarkMetric:
        input_ids = torch.ones(size=(batch_size, ), dtype=torch.int, device=self.device)
        image_features = None
        position_ids = tensor = torch.full((batch_size,), n_kv_cache_tokens, dtype=torch.int, device=self.device)
        selected_token_ids = torch.tensor(list(range(batch_size)), dtype=torch.int, device=self.device)
        n_layers = self.language_config.n_layers
        n_tokens = 2
        block_size = 16
        # to analsis large batchsize, we share same kv cache for all requests
        n_blocks = (batch_size + n_kv_cache_tokens + block_size - 1) // block_size
        n_kv_heads = self.language_config.n_kv_heads
        head_size = self.language_config.head_dim
        builder = AttentionParametersBuilder(
            num_qo_heads=self.language_config.n_qo_heads, 
            num_kv_heads=self.language_config.n_kv_heads, 
            head_dim=self.language_config.head_dim, 
            block_size=block_size, 
            device=self.device, 
        )
        kv_cache = torch.randn(size=(n_layers, n_tokens, n_blocks, block_size, n_kv_heads, head_size), dtype=self.dtype, device=self.device)
        for i in range(self.language_config.n_layers):
            layer_cache = KVCache(
                key_cache=kv_cache[i, 0, :, :, :, :],
                value_cache=kv_cache[i, 1, :, :, :, :]
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
        model_params = LanguageModelParameters(
            input_ids_or_input_embeds=input_ids, 
            position_ids=position_ids, 
            image_features=None, 
            image_overwrite_mask=None, 
            attention_params=builder.build_attention_parameters(), 
            all_sequences_decode=False, 
            selected_token_ids=selected_token_ids
        )
        def forward():
            self.language_model.forward(input_ids=input_ids, image_features=image_features, position_ids=position_ids, model_params=model_params)
        latency = benchmark(forward, n_iter=20)
        throughput = batch_size / latency
        metric = BatchSizeBenchmarkMetric(
            name = 'decode', 
            batch_size = batch_size, 
            latency = latency, 
            throughput = throughput, 
        )
        logger.info(f'batch profile result {metric}')
        return metric

    def profile_decode_list(self, batch_sizes: list[int], n_kv_cache_tokens: int) -> list[BatchSizeBenchmarkMetric]:
        return [self.profile_decode(batch_size, n_kv_cache_tokens) for batch_size in  batch_sizes]

    def profile(
        self,
        result_path: Optional[str]=None, 
        encode_batch_sizes: list[int] = [1, 2, 3, 4, 5, 6, 7, 8], 
        prefill_batch_sizes: list[int] = [1, 2, 3, 4, 5, 6, 7, 8], 
        n_prompt_tokens: int = 1024, 
        decode_batch_sizes: list[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048], 
        n_kv_cache_tokens: int = 1024, 
    ) -> BatchSizeAnalysisResult:
        result = BatchSizeAnalysisResult(
            encode=self.profile_encode_list(encode_batch_sizes), 
            prefill=self.profile_prefill_list(prefill_batch_sizes, n_prompt_tokens), 
            decode=self.profile_decode_list(decode_batch_sizes, n_kv_cache_tokens), 
        )
        if result_path:
            with open(result_path, "w") as file:
                json.dump(asdict(result), fp=file, indent=4)
        return result

    def plot(
        self, 
        result_path: str, 
        fig_path: str, 
    ):
        if not os.path.exists(result_path):
            logger.info(f'{result_path} not exists')
            return
        from hydrainfer.utils.json_utils import load_json
        result = load_json(data_class=BatchSizeAnalysisResult, data_path=result_path)
            
        @dataclass
        class BarData:
            name: str
            batch_size: list[int]
            latencies: list[float]
            throughputs: list[float]
        
        all_bar_data: list[BarData] = []
        for stage_data in [result.encode, result.prefill, result.decode]:
            name = stage_data[0].name
            batch_size_list = []
            latency_list = []
            throughput_list = []
            for batch_size_data in stage_data:
                batch_size_list.append(batch_size_data.batch_size)
                latency_list.append(batch_size_data.latency)
                throughput_list.append(batch_size_data.throughput)
                
            all_bar_data.append(BarData(
                name=name,  
                batch_size=batch_size_list, 
                latencies=latency_list, 
                throughputs=throughput_list, 
            ))

        n_rows = 3
        n_cols = 2
        name_id = {'encode' : 0, 'prefill' : 1, 'decode' : 2}
        figsize=(3.5 * n_cols, 3.5 * n_rows)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.atleast_2d(axes)
        plt.subplots_adjust(hspace=0.5, wspace=0.5) 

        def to_k_format(x, pos):
            if x >= 1000:
                return f'{x/1000:.0f}K' if x % 1000 == 0 else f'{x/1000:.1f}K'
            return str(int(x))

        fontsize=14
        formatter = FuncFormatter(to_k_format)
        for bar_data in all_bar_data:
            i = name_id[bar_data.name]
            n_ticks = len(bar_data.batch_size)
            x_ticks = [i for i in range(n_ticks)]
            x_ticklabels = [i for i in bar_data.batch_size]
            width=0.6
            axes[i][0].bar(x_ticks, bar_data.throughputs, width=width, color="#6EA6A9", edgecolor='black', alpha=1)
            axes[i][0].set_xlabel('Batch Size', fontsize=fontsize)
            axes[i][0].set_ylabel('Throughput (token/s)', fontsize=fontsize)
            axes[i][0].yaxis.set_major_formatter(formatter)
            
            axes[i][1].bar(x_ticks, bar_data.latencies, width=width, color="#6EA6A9", edgecolor='black', alpha=1)
            axes[i][1].set_xlabel('Batch Size', fontsize=fontsize)
            axes[i][1].set_ylabel('Latency (s)', fontsize=fontsize)
            
            for j in range(n_cols):
                axes[i][j].tick_params(axis='x', which='major', length=2, width=1, direction='out', grid_color='black', grid_alpha=1)
                axes[i][j].tick_params(axis='y', which='major', length=2, width=1, direction='out', grid_color='black', grid_alpha=1)
                axes[i][j].tick_params(which='both', bottom=True, top=False, left=True, right=False, labelbottom=True, labelleft=True, direction='out')
                axes[i][j].set_title(bar_data.name, fontsize=fontsize, color='black', alpha=1)
                axes[i][j].grid(False)
                axes[i][j].set_xticks(x_ticks)
                axes[i][j].set_xticklabels(x_ticklabels, fontsize=fontsize - 5, rotation=45, color='black', alpha=1)
                for label in axes[i][j].get_yticklabels():
                    label.set_fontsize(fontsize - 5)
                for spine in axes[i][j].spines.values():
                    spine.set_edgecolor('black')
            
            fig.savefig(fig_path, bbox_inches="tight")
