import time
import torch
import argparse
from torch import Tensor
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields
from transformers import AutoTokenizer, AutoProcessor
from dxz.request.rcb import RequestControlBlock
from dxz.engine.isa import Instruction, ImageFill, ImageEmbedFill, Fill, EmptyInstruction
from dxz.model.parameters import AttentionParametersBuilder, LanguageModelParameters, VisionModelParameters
from dxz.model.model_factory import VisionModelConfig, LanguageModelConfig, VisionModel, LanguageModel
from dxz.memory.virtual_kv_cache import MemoryManagementUnit


@dataclass
class ExecutorConfig:
    use_flash_infer: bool = True
    batch_image_embed_forward: bool = True
    multi_streams_forward: bool = False
    multi_threads_forward: bool = False

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> 'ExecutorConfig':
        attrs = [attr.name for attr in fields(cls)]
        config = cls(**{attr: getattr(args, attr) for attr in attrs})
        return config

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--use-flash-infer', action='store_true', help='Enable flash infer attention kernel, default kernel is flash attention')
        parser.add_argument('--batch-image-embed-forward', action='store_true', help='Enable batch image embedding forwarding.')
        parser.add_argument('--multi-streams-forward', action='store_true', help='Enable multi-stream forwarding.')
        parser.add_argument('--multi-threads-forward', action='store_true', help='Enable multi-thread forwarding.')
        return parser


@dataclass
class ExecutorContext:
    vision_model: VisionModel
    vision_model_config: VisionModelConfig
    language_model: LanguageModel
    language_model_config: LanguageModelConfig
    processor: AutoProcessor
    tokenizer: AutoTokenizer
    dtype: torch.dtype
    device: torch.device
    block_size: int
    mmu: MemoryManagementUnit


class Executor:
    def execute(self, contexts: list[tuple[RequestControlBlock, Instruction]]):
        raise Exception('interface not implemented')


class BatchFillExecutor(Executor):
    def __init__(self, config: ExecutorConfig, context: ExecutorContext):
        self.config = config
        self.context = context
        self.dtype = context.dtype
        self.device = context.device
        self.vision_model = context.vision_model
        self.vision_model_config = context.vision_model_config
        self.language_model = context.language_model
        self.language_model_config = context.language_model_config
        self.mmu = context.mmu

        self.workspace_buffer = None
        self.batch_prefill_with_paged_kvcache_wrapper = None
        self.batch_decode_with_paged_kvcache_wrapper = None
        if config.use_flash_infer:
            import flashinfer
            self.workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
            self.batch_prefill_with_paged_kvcache_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.workspace_buffer, "NHD")
            self.batch_decode_with_paged_kvcache_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(self.workspace_buffer, "NHD")

    def execute(self, contexts: list[tuple[RequestControlBlock, Instruction]]):
        if len(contexts) == 0:
            return {}

        pixel_values: list[Tensor] = []
        image_features: list[Tensor] = []
        for seq, inst in contexts:
            if isinstance(inst, ImageFill):
                pixel_values.append(inst.pixel_values)
                inst.pixel_values = None
            if isinstance(inst, ImageEmbedFill):
                image_features.append(inst.image_features)
                inst.image_features = None
        if len(pixel_values) > 0 and len(image_features) > 0:
            raise Exception('not support pixel value and image embed batch')

        if len(pixel_values):
            pixel_values = torch.cat(pixel_values, dim=0).to(dtype=self.dtype, device=self.device)
        else:
            pixel_values = None

        if len(image_features):
            image_features = torch.cat(image_features, dim=0).to(dtype=self.dtype, device=self.device)
        else:
            image_features = None

        token_ids         : list[int] = []
        position_ids      : list[int] = []
        selected_token_ids: list[int] = []
        attention_params_builder = AttentionParametersBuilder(
            num_qo_heads = self.language_model_config.n_qo_heads,
            num_kv_heads = self.language_model_config.n_kv_heads,
            head_dim = self.language_model_config.head_dim, 
            block_size = self.context.block_size, 
            device = self.device, 
            flash_infer_batch_prefill_handler = self.batch_prefill_with_paged_kvcache_wrapper, 
            flash_infer_batch_decode_handler = self.batch_decode_with_paged_kvcache_wrapper, 
        )
        for seq, inst in contexts:
            virtual_kv_cache = seq.virtual_kv_caches[inst.kv_cache_ids[0]]
            virtual_kv_cache.set(inst.cache_ids[0])
            slot_ids = virtual_kv_cache.v2p(inst.cache_ids[0])

            token_ids += inst.token_ids
            position_ids += inst.position_ids
            if inst.sample:
                selected_token_ids.append(len(token_ids) - 1)
            attention_params_builder.add_request(
                q_seq_len = len(inst.token_ids), 
                kv_seq_len = virtual_kv_cache.n_kv_cache_tokens, 
                new_cache_slots = slot_ids, 
                block_table = virtual_kv_cache.block_table
            )
        for layer_id in range(self.language_model_config.n_layers):
            attention_params_builder.add_kv_cache(self.mmu.kv_caches[layer_id])
        layers_attention_params = attention_params_builder.build_attention_parameters()

        model_params = LanguageModelParameters(
            attention_params = layers_attention_params, 
            all_sequences_decode = layers_attention_params[0].all_sequences_decode, 
            selected_token_ids = selected_token_ids
        )
        ten_input_ids = torch.tensor(token_ids, dtype=torch.int, device=self.device)
        ten_position_ids = torch.tensor(position_ids, dtype=torch.int, device=self.device)

        if pixel_values is not None:
            image_features = self.vision_model.forward(pixel_values, VisionModelParameters(return_last_layer_attention=False)).image_features
        sample_token_ids = self.language_model.forward(ten_input_ids, image_features, ten_position_ids, model_params).sample_token_ids
        sample_token_ids = sample_token_ids.tolist()

        if len(selected_token_ids) > 0:
            t = time.perf_counter()
            i = 0
            for rcb, instruction in contexts:
                if (isinstance(instruction, Fill)) and instruction.sample:
                    next_token_id = sample_token_ids[i]
                    instruction.sample_dst.token_ids = [next_token_id]
                    is_last_token =  isinstance(instruction.sample_dst, EmptyInstruction)
                    rcb.metric.tokens_time.append(t)
                    rcb.output_token_processor.append_token_id(next_token_id, is_last_token)
                    i += 1

        for rcb, _ in contexts:
            rcb.pc += 1

class ImageEmbedExecutor(Executor):
    def __init__(self, context: ExecutorContext):
        self.context = context

    def execute(self, contexts: list[tuple[RequestControlBlock, Instruction]]):
        if len(contexts) == 0:
            return
        for rcb, instruction in contexts:
            pixel_values = instruction.pixel_values.to(self.context.dtype).to(self.context.device)
            vision_params = VisionModelParameters(return_last_layer_attention=False)
            image_features = self.vision_model.forward(pixel_values, vision_params).image_features
            instruction.image_featues_dst.image_features = image_features

        for rcb, _ in contexts:
            rcb.pc += 1

        return


class BatchImageEmbedExecutor(Executor):
    def __init__(self, context: ExecutorContext):
        self.context = context
        self.vision_model = context.vision_model

    def execute(self, contexts: list[tuple[RequestControlBlock, Instruction]]):
        if len(contexts) == 0:
            return
        n_images: list[int] = []
        batch_pixel_values: list[Tensor] = []
        for rcb, instruction in contexts:
            pixel_values = instruction.pixel_values.to(self.context.dtype).to(self.context.device) # (n_images, n_channels, width, height)
            batch_pixel_values.append(pixel_values)
            n_images.append(pixel_values.shape[0])
        pixel_values = torch.cat(batch_pixel_values, dim=0) 

        vision_params = VisionModelParameters(return_last_layer_attention=False)
        image_features = self.vision_model.forward(pixel_values, vision_params).image_features

        left = 0
        for i, (rcb, instruction) in enumerate(contexts):
            right = left + n_images[i]
            instruction.image_featues_dst.image_features = image_features[left: right, :, :]
            left += n_images[i]

        for rcb, _ in contexts:
            rcb.pc += 1


class MultiStreamsDecorator(Executor):
    def __init__(self, stream: torch.cuda.Stream, executor: Executor):
        self.stream = stream
        self.executor = executor

    def execute(self, contexts: list[tuple[RequestControlBlock, Instruction]]):
        if len(contexts) == 0:
            return
        with torch.cuda.stream(self.stream):
            self.executor.execute(contexts)
            self.stream.synchronize()


class MultiThreadsDecorator:
    def __init__(self, pool: ThreadPoolExecutor, executor: Executor):
        self.pool = pool
        self.executor = executor

    def execute(self, contexts: list[tuple[RequestControlBlock, Instruction]]):
        if len(contexts) == 0:
            return

        future = self.pool.submit(self.executor.execute, contexts)
        return future


class InstructionExecutor:
    def __init__(self, config: ExecutorConfig, context: ExecutorContext):
        self.config = config
        self.context = context
        self.fill_executor = BatchFillExecutor(config, context)

        if self.config.batch_image_embed_forward:
            self.image_embed_executor = BatchImageEmbedExecutor(context)
        else:
            self.image_embed_executor = ImageEmbedExecutor(context)

        self.streams: list[torch.cuda.Stream] = []
        if self.config.multi_streams_forward:
            stream = torch.cuda.Stream()
            self.streams.append(stream)
            self.image_embed_executor = MultiStreamsDecorator(stream, self.image_embed_executor)

        self.pool: Optional[ThreadPoolExecutor] = None
        if self.config.multi_threads_forward:
            self.pool = ThreadPoolExecutor(max_workers=1)
            self.image_embed_executor = MultiThreadsDecorator(self.pool, self.image_embed_executor)

    def execute_fill(self, contexts: list[tuple[RequestControlBlock, Instruction]]):
        return self.fill_executor.execute(contexts)

    def execute_image_embed(self, contexts: list[tuple[RequestControlBlock, Instruction]]):
        return self.image_embed_executor.execute(contexts)

    def execute_empty(self, contexts: list[tuple[RequestControlBlock, Instruction]]):
        for rcb, _ in contexts:
            rcb.pc += 1