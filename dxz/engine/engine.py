from concurrent.futures import ThreadPoolExecutor
import time
import random
from typing import Optional
from itertools import accumulate
from transformers import AutoTokenizer, AutoProcessor
from dataclasses import dataclass, field, fields
import torch
from torch import Tensor
from dxz.engine.executor import Executor, ExecutorConfig
from dxz.engine.isa import Instruction, Fill, TextFill, ImageFill, Mov, ReAlloc, EmptyInstruction, ImageEmbedFill, ImageEmbed
from dxz.engine.request import Request
from dxz.model.downloader import download_hf_model
from dxz.model.parameters import AttentionParameters, LanguageModelParameters, VisionModelParameters, AttentionParametersBuilder
from dxz.request.rcb import RequestControlBlock
from dxz.request.request_processor import RequestProcessorConfig, RequestProcessor, RequestProcessorContext, RequestProcessParameters
from dxz.memory.virtual_kv_cache import VirtualKVCache, MemoryManagementUnit, MemoryConfig, MemoryContext
from dxz.engine.scheduler import SchedulerConfig, RequestScheduler
from dxz.model.model_factory import ModelFactory
from dxz.engine.executor import InstructionExecutor, ExecutorContext, ExecutorConfig
import argparse

@dataclass
class EngineConfig:
    model_name: str          = "llava-hf/llava-1.5-7b-hf" 
    model_path: Optional[str]= None
    dtype: torch.dtype  = torch.half 
    device: torch.device = torch.device('cuda:0') 
    memory_config: MemoryConfig = field(default_factory=MemoryConfig)
    request_processor_config: RequestProcessorConfig = field(default_factory=RequestProcessorConfig)
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    executor_config: ExecutorConfig = field(default_factory=ExecutorConfig)
    multi_thread_request_process: bool = True
    warm_up: bool = True

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> 'EngineConfig':
        attrs = [attr.name for attr in fields(cls) if attr.name not in ['dtype', 'device', 'memory_config', 'request_processor_config', 'scheduler_config', 'executor_config']]
        memory_config = MemoryConfig.from_cli_args(args)
        request_processor_config = RequestProcessorConfig.from_cli_args(args)
        scheduler_config = SchedulerConfig.from_cli_args(args)
        executor_config = ExecutorConfig.from_cli_args(args)
        config = cls(memory_config=memory_config, request_processor_config=request_processor_config, scheduler_config=scheduler_config, executor_config=executor_config, **{attr: getattr(args, attr) for attr in attrs})
        return config

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = MemoryConfig.add_cli_args(parser)
        parser = RequestProcessorConfig.add_cli_args(parser)
        parser = SchedulerConfig.add_cli_args(parser)
        parser = ExecutorConfig.add_cli_args(parser)
        parser.add_argument('--model-name', type=str, default="llava-hf/llava-1.5-7b-hf", help='The name of the model.')
        parser.add_argument('--model_path', type=str, nargs="?", default=None, help="path to the model, if set none will download model from huggingface to default cache directory of transformers library with the model-name arg.")
        parser.add_argument('--multi-thread-request-process', action='store_true', help='Enable multi-threading for request processing.')
        parser.add_argument('--warm-up', action='store_true', help='Enable warm-up phase.')
        return parser


class Engine:
    def __init__(self, config: EngineConfig):
        self.config = config
        # 1. model
        model_factory = ModelFactory(config.model_name, config.model_path, config.dtype, config.device)
        self.vision_model, self.vision_model_config = model_factory.getVisionModel() 
        self.language_model, self.language_model_config = model_factory.getLanguageModel() 
        self.processor = model_factory.getProcessor() 
        self.tokenizer = model_factory.getTokenizer() 

        # 2. memory
        self.memory_context = MemoryContext(
            n_layers=self.language_model_config.n_layers,
            head_size = self.language_model_config.head_dim, 
            num_kv_heads=self.language_model_config.n_kv_heads, 
            dtype = self.config.dtype, 
            device = self.config.device, 
        )
        self.mmu = MemoryManagementUnit(
            config = self.config.memory_config, 
            context = self.memory_context, 
            )

        # 3. executor
        executor_context = ExecutorContext(
            vision_model = self.vision_model, 
            vision_model_config = self.vision_model_config, 
            language_model = self.language_model, 
            language_model_config = self.language_model_config, 
            processor = self.processor, 
            tokenizer = self.tokenizer, 
            dtype = config.dtype, 
            device = config.device, 
            block_size = config.memory_config.block_size, 
            mmu = self.mmu
        )
        self.executor = InstructionExecutor(config.executor_config, executor_context)

        # 3. scheduler
        self.scheduler = RequestScheduler(self.config.scheduler_config)

        # 4. model warm up optimization
        if config.warm_up:
            self.warm_up()

    def warm_up(self):
        n_tokens = 596
        n_blocks = (n_tokens + self.config.memory_config.block_size - 1) // self.config.memory_config.block_size
        params = LanguageModelParameters(
            attention_params=[AttentionParameters(
                kv_cache = self.mmu.kv_caches[0], 
                q_cu_seq_lens = torch.tensor([0, n_tokens], dtype=torch.int, device=self.config.device), 
                kv_cu_seq_lens = torch.tensor([0, n_tokens], dtype=torch.int, device=self.config.device), 
                paged_kv_last_page_len = None, 
                new_cache_slots = torch.arange(n_tokens, dtype=torch.int, device=self.config.device),
                block_tables = torch.arange(n_blocks, dtype=torch.int, device=self.config.device), 
                cu_blocks_lens = torch.tensor([0, n_blocks], dtype=torch.int, device=self.config.device), 
                num_sequences = 1, 
                all_sequences_decode = False, 
                q_max_seq_len = n_tokens, 
                kv_max_seq_len = n_tokens, 
            ) for _ in range(self.language_model_config.n_layers)],
            all_sequences_decode=False,  
            selected_token_ids=torch.arange(n_tokens, dtype=torch.int, device=self.config.device)
        )
        input_ids = torch.zeros(n_tokens, dtype=torch.int, device=self.config.device)
        position_ids = torch.arange(n_tokens, dtype=torch.int, device=self.config.device)
        image_features = None
        for i in range(3):
            self.language_model.forward(input_ids, image_features, position_ids, params)
    
    @torch.inference_mode()
    def step(self) -> dict[int, int]:
        # 1. schedule requests
        contexts = self.scheduler.step()
        if len(contexts) == 0:
            return {}

        # 2. execute instructions
        fill_contexts = []
        image_embed_contexts = []
        for context in contexts:
            rcb, instruction = context
            if isinstance(instruction, Fill):
                fill_contexts.append(context)
                continue
            if isinstance(instruction, Mov):
                raise Exception('not implemented')
                continue
            if isinstance(instruction, ReAlloc):
                self.execute_realloc(context)
                continue
            if isinstance(instruction, EmptyInstruction):
                continue
            if isinstance(instruction, ImageEmbed):
                image_embed_contexts.append(context)
                continue
            raise Exception(f'unsupported instrction {type(instruction)}')

        future = self.executor.execute_image_embed(image_embed_contexts)
        output_tokens = self.executor.execute_fill(fill_contexts)
        if future is not None:
            future.result()

        # 3. scheduler requests
        t = time.perf_counter()
        for rcb, _ in contexts:
            if rcb.is_finished():
                rcb.metric.finished_time = t
                self.scheduler.schedule_finished([rcb])
            else:
                self.scheduler.schedule_unfinished([rcb])
        
        return output_tokens