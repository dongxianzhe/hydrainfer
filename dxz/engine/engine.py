import io
import base64
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import random
from itertools import accumulate
from transformers import AutoTokenizer, AutoProcessor
from PIL import Image
from dataclasses import dataclass, field, fields
import torch
from torch import Tensor
from dxz.engine.isa import Instruction, Fill, TextFill, ImageFill, Mov, ReAlloc, EmptyInstruction, ImageEmbedFill, ImageEmbed
from dxz.engine.request import Request
from dxz.model.downloader import download_hf_model
from dxz.model.llava import LlavaForConditionalGeneration
from dxz.model.parameters import AttentionParameters, ModelParameters, VisionModelParameters, AttentionParametersBuilder
from dxz.sequence.sequence import Sequence
from dxz.memory.compiler import CompilerConfig, Compiler, CompilerContext, CompileParameters
from dxz.memory.virtual_kv_cache import VirtualKVCache, MemoryManagementUnit, MemoryConfig, MemoryContext
from dxz.engine.scheduler import SchedulerConfig, SequenceScheduler
import argparse

@dataclass
class EngineConfig:
    model_name  : str          = "llava-hf/llava-1.5-7b-hf" 
    dtype       : torch.dtype  = torch.half 
    device      : torch.device = torch.device('cuda:0') 
    memory_config    : MemoryConfig    = field(default_factory=MemoryConfig)
    compiler_config  : CompilerConfig  = field(default_factory=CompilerConfig)
    scheduler_config : SchedulerConfig = field(default_factory=SchedulerConfig)
    multi_thread_request_process : bool = True
    batch_image_embed_forward: bool = True
    multi_streams_forward: bool = False
    multi_threads_forward: bool = False
    warm_up: bool = True

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> 'MemoryConfig':
        attrs = [attr.name for attr in fields(cls) if attr.name not in ['model_name', 'dtype', 'device', 'memory_config', 'compiler_config', 'scheduler_config']]
        memory_config = MemoryConfig.from_cli_args(args)
        compiler_config = CompilerConfig.from_cli_args(args)
        scheduler_config = SchedulerConfig.from_cli_args(args)
        config = cls(memory_config=memory_config, compiler_config=compiler_config, scheduler_config=scheduler_config, **{attr: getattr(args, attr) for attr in attrs})
        return config

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = MemoryConfig.add_cli_args(parser)
        parser = CompilerConfig.add_cli_args(parser)
        parser = SchedulerConfig.add_cli_args(parser)
        parser.add_argument('--model-name', type=str, default="llava-hf/llava-1.5-7b-hf", help='The name of the model.')
        parser.add_argument('--multi-thread-request-process', action='store_true', help='Enable multi-threading for request processing.')
        parser.add_argument('--batch-image-embed-forward', action='store_true', help='Enable batch image embedding forwarding.')
        parser.add_argument('--multi-streams-forward', action='store_true', help='Enable multi-stream forwarding.')
        parser.add_argument('--multi-threads-forward', action='store_true', help='Enable multi-thread forwarding.')
        parser.add_argument('--warm-up', action='store_true', help='Enable warm-up phase.')
        return parser


class Engine:
    def __init__(self, config: EngineConfig):
        self.config = config
        # 1. model
        model_path = download_hf_model(repo_id=config.model_name)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer
        self.model = LlavaForConditionalGeneration.from_safetensor(model_path, self.config.dtype, self.config.device)
        self.model_config = self.model.config
        self.model_runner = self.model
        # 2. memory
        self.memory_context = MemoryContext(
            n_layers=self.model_config.text_config.num_hidden_layers,
            head_size = self.model_config.text_config.head_dim, 
            num_kv_heads=self.model_config.text_config.num_key_value_heads, 
            dtype = self.config.dtype, 
            device = self.config.device, 
        )
        self.mmu = MemoryManagementUnit(
            config = self.config.memory_config, 
            context = self.memory_context, 
            )
        # 3. compiler
        self.compiler_context = CompilerContext(
            tokenizer = self.tokenizer, 
            processor = self.processor, 
            image_token_id = self.model_config.image_token_index, 
            num_image_tokens = self.model_config.image_seq_length, 
            n_layers = self.model_config.text_config.num_hidden_layers,
        )
        self.compiler = Compiler(
            config = self.config.compiler_config, 
            context = self.compiler_context, 
        )
        # 4. sequence
        self.sid_allocator = 0
        self.scheduler = SequenceScheduler(self.config.scheduler_config)

        # 5. flashinfer optimization
        import flashinfer
        self.workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.config.device)
        self.batch_prefill_with_paged_kvcache_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.workspace_buffer, "NHD")
        self.batch_decode_with_paged_kvcache_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(self.workspace_buffer, "NHD")

        # 6. multi threads request process optimization
        self.add_request_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=32)

        # 7. multi stream and threads forward optimization
        self.streams: list[torch.cuda.Stream] = [torch.cuda.Stream(), torch.cuda.Stream()]
        self.forward_executor = ThreadPoolExecutor(max_workers=2)

        # 8. model warm up optimization
        if config.warm_up:
            self.warm_up()

    def warm_up(self):
        n_tokens = 596
        n_blocks = (n_tokens + self.config.memory_config.block_size - 1) // self.config.memory_config.block_size
        params = ModelParameters(
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
            ) for _ in range(self.model_config.text_config.num_hidden_layers)],
            all_sequences_decode=False,  
        )
        input_ids = torch.zeros(n_tokens, dtype=torch.int, device=self.config.device)
        position_ids = torch.arange(n_tokens, dtype=torch.int, device=self.config.device)
        pixel_values = None
        image_features = None
        for i in range(3):
            self.model(input_ids, pixel_values, image_features, position_ids, params)


    def add_request(self, request: Request):
        if self.config.multi_thread_request_process:
            self.executor.map(self._add_request_async, [request])
        else:
            self._add_request(request)

    def _add_request_async(self, request: Request):
        with self.add_request_lock:
            self._add_request(request)
    
    def _add_request(self, request: Request):
        if request.image is None and request.image_base64 is not None:
            request.image = Image.open(io.BytesIO(base64.b64decode(request.image_base64)))
        arrival_time = time.perf_counter()

        static_info = self.compiler.compile(
            prompt = request.prompt, 
            images = request.image, 
            compile_params = CompileParameters(max_tokens = request.max_tokens)
            )
        sequence = Sequence(
            static_info=static_info, 
            sid = self.sid_allocator, 
            instructions = static_info.instructions, 
            virtual_kv_caches = self.mmu.allocate_virtual_kv_caches(static_info.n_virtual_kv_caches), 
            max_tokens = request.max_tokens, 
            eos_token_id = None, 
            max_seq_len = self.model_config.text_config.max_position_embeddings, 
            rid = request.request_id, 
        )
        sequence.metric.arrival_time = arrival_time
        self.sid_allocator += 1
        self.scheduler.schedule_new([sequence])
    
    def execute_batch_fill(self, contexts: list[tuple[Sequence, Instruction]]) -> dict[int, int]:
        if len(contexts) == 0:
            return {}

        pixel_values: list[Tensor] = []
        image_features: list[Tensor] = []
        has_image_fill: bool = False
        has_image_embed_fill: bool = False
        for seq, inst in contexts:
            if isinstance(inst, ImageFill):
                pixel_values.append(inst.pixel_values)
                inst.pixel_values = None
                has_image_fill = True
            if isinstance(inst, ImageEmbedFill):
                image_features.append(inst.image_features)
                inst.image_features = None
                has_image_embed_fill = True
        if has_image_fill and has_image_embed_fill:
            raise Exception('not support pixel value and image embed batch')
        if len(pixel_values):
            pixel_values = torch.cat(pixel_values, dim=0).to(dtype=self.config.dtype, device=self.config.device)
        else:
            pixel_values = None
        if len(image_features):
            image_features = torch.cat(image_features, dim=0).to(dtype=self.config.dtype, device=self.config.device)
        else:
            image_features = None

        token_ids         : list[int] = []
        position_ids      : list[int] = []
        selected_token_ids: list[int] = []
        for seq, inst in contexts:
            token_ids += inst.token_ids
            position_ids += inst.position_ids
            if inst.sample:
                selected_token_ids.append(len(token_ids) - 1)
        
        attention_params_builder = AttentionParametersBuilder(
            num_qo_heads = self.model_config.text_config.num_attention_heads,
            num_kv_heads = self.model_config.text_config.num_key_value_heads,
            head_dim = self.model_config.text_config.head_dim, 
            block_size = self.config.memory_config.block_size, 
            device = self.config.device, 
            flash_infer_batch_prefill_handler = self.batch_prefill_with_paged_kvcache_wrapper, 
            flash_infer_batch_decode_handler = self.batch_decode_with_paged_kvcache_wrapper, 
        )
        for seq, inst in contexts:
            virtual_kv_cache = seq.virtual_kv_caches[inst.kv_cache_ids[0]]
            virtual_kv_cache.set(inst.cache_ids[0])
            slot_ids = virtual_kv_cache.v2p(inst.cache_ids[0])
            attention_params_builder.add_request(
                q_seq_len = len(inst.token_ids), 
                kv_seq_len = virtual_kv_cache.n_kv_cache_tokens, 
                new_cache_slots = slot_ids, 
                block_table = virtual_kv_cache.block_table
            )
        for layer_id in range(self.model_config.text_config.num_hidden_layers):
            attention_params_builder.add_kv_cache(self.mmu.kv_caches[layer_id])
        layers_attention_params = attention_params_builder.build_attention_parameters()

        model_params = ModelParameters(
            attention_params = layers_attention_params, 
            all_sequences_decode = layers_attention_params[0].all_sequences_decode, 
            selected_token_ids = selected_token_ids
        )
        ten_input_ids = torch.tensor(token_ids, dtype=torch.int, device=self.config.device)
        ten_position_ids = torch.tensor(position_ids, dtype=torch.int, device=self.config.device)

        sample_token_ids = self.model_runner(ten_input_ids, pixel_values, image_features, ten_position_ids, model_params)
        sample_token_ids = sample_token_ids.tolist()
        output_tokens = {} # rid -> token_id
        if len(selected_token_ids) > 0:
            t = time.perf_counter()
            i = 0
            for sequence, instruction in contexts:
                if (isinstance(instruction, Fill)) and instruction.sample:
                    next_token_id = sample_token_ids[i]
                    instruction.sample_dst.token_ids = [next_token_id]
                    sequence.output_token_ids.append(next_token_id)
                    sequence.metric.tokens_time.append(t)
                    i += 1
                    output_tokens[sequence.rid] = next_token_id
        return output_tokens

    def execute_mov(self, context: tuple[Sequence, Instruction]):
        raise Exception('not implemented')
        sequence, instruction = context
        src_slot_ids: list[int] = []
        for src_cache_ids, src_kv_cache_id in zip(instruction.src_cache_ids, instruction.src_kv_cache_ids):
            block_table = sequence.virtual_kv_caches[src_kv_cache_id].block_tables
            src_slot_ids += self.mmu.v2p(src_cache_ids, block_table)
        dst_slot_ids: list[int] = []
        for dst_cache_ids, dst_kv_cache_id in zip(instruction.dst_cache_ids, instruction.dst_kv_cache_ids):
            block_table = sequence.virtual_kv_caches[dst_kv_cache_id].block_tables
            dst_slot_ids += self.mmu.v2p(dst_cache_ids, block_table)
        assert len(src_slot_ids) == len(dst_slot_ids), f'{len(src_slot_ids)} {len(dst_slot_ids)}'
        src_slot_ids = torch.tensor(src_slot_ids, dtype=torch.int, device=self.config.device)
        dst_slot_ids = torch.tensor(dst_slot_ids, dtype=torch.int, device=self.config.device)
        self.mmu.move_physical_kv_caches(src_slot_ids, dst_slot_ids)


    def execute_realloc(self, context: tuple[Sequence, Instruction]):
        sequence, instruction = context
        for n_token, vid in zip(instruction.n_tokens, instruction.kv_cache_ids):
            sequence.virtual_kv_caches[vid].realloc(n_token)

    def execute_batch_image_embed(self, contexts: list[tuple[Sequence, Instruction]]):
        if len(contexts) == 0:
            return

        n_images: list[int] = []
        batch_pixel_values: list[Tensor] = []
        for sequence, instruction in contexts:
            pixel_values = instruction.pixel_values.to(self.config.dtype).to(self.config.device) # (n_images, n_channels, width, height)
            batch_pixel_values.append(pixel_values)
            n_images.append(pixel_values.shape[0])
        pixel_values = torch.cat(batch_pixel_values, dim=0) 

        model_params = ModelParameters(
            embed_token_pruning_params=instruction.token_pruning_params, 
            vision_params=VisionModelParameters(
                return_last_layer_attention=True if instruction.token_pruning_params.get('policy', None) else None
            )
        )
        image_features = self.model.image_embed(pixel_values, model_params) # (batch_size, n_tokens, vision_hidden_size)

        left = 0
        for i, (sequence, instruction) in enumerate(contexts):
            right = left + n_images[i]
            instruction.image_featues_dst.image_features = image_features[left: right, :, :]
            left += n_images[i]

    def execute_batch_image_embed_streams_decorator(self, contexts: list[tuple[Sequence, Instruction]], stream: torch.cuda.Stream):
        with torch.cuda.stream(stream):
            self.execute_batch_image_embed(contexts)
            stream.synchronize()

    def execute_image_embed(self, context: tuple[Sequence, Instruction]):
        sequence, instruction = context
        pixel_values = instruction.pixel_values.to(self.config.dtype).to(self.config.device)
        model_params = ModelParameters(
            embed_token_pruning_params=instruction.token_pruning_params, 
            vision_params=VisionModelParameters(
                return_last_layer_attention=True if instruction.token_pruning_params else None
            )
        )
        image_features = self.model.image_embed(pixel_values, model_params)
        instruction.image_featues_dst.image_features = image_features

    @torch.inference_mode()
    def step(self) -> dict[int, int]:
        # 1. schedule sequence
        contexts = self.scheduler.step()
        if len(contexts) == 0:
            return {}

        # 2. execute instructions
        fill_contexts = []
        image_embed_contexts = []
        for context in contexts:
            sequence, instruction = context
            if isinstance(instruction, Fill):
                fill_contexts.append(context)
                continue
            if isinstance(instruction, Mov):
                self.execute_mov(context)
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

        if self.config.multi_threads_forward:
            if(len(image_embed_contexts) > 0):
                if self.config.multi_streams_forward:
                    embed_future = self.forward_executor.submit(self.execute_batch_image_embed_streams_decorator, image_embed_contexts, self.streams[1])
                else:
                    embed_future = self.forward_executor.submit(self.execute_batch_image_embed, image_embed_contexts)
            output_tokens = self.execute_batch_fill(fill_contexts)
            if(len(image_embed_contexts) > 0):
                embed_future.result()
        else:
            output_tokens = self.execute_batch_fill(fill_contexts)
            if self.config.batch_image_embed_forward:
                if self.config.multi_streams_forward:
                    self.execute_batch_image_embed_streams_decorator(image_embed_contexts, self.streams[1])
                else:
                    self.execute_batch_image_embed(image_embed_contexts)
            else:
                for context in image_embed_contexts:
                    self.execute_image_embed(context)

        # 3. scheduler sequence
        t = time.perf_counter()
        for sequence, _ in contexts:
            if sequence.is_finished():
                sequence.metric.finished_time = t
                self.scheduler.schedule_finished([sequence])
            else:
                self.scheduler.schedule_unfinished([sequence])
        
        return output_tokens