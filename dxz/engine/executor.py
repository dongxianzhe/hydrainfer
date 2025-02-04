import time
import torch
import argparse
from torch import Tensor
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields, field
from dxz.request.rcb import RequestControlBlock
from dxz.engine.isa import Instruction, ImageFill, ImageEmbedFill, Fill, EmptyInstruction
from dxz.layer.causal_attention import AttentionParametersBuilder, AttentionParameters
from dxz.model.parameters import LanguageModelParameters, VisionModelParameters
from dxz.model.model_factory import ModelFactory, getModelFactory, ModelFactoryConfig, ModelFactoryContext
from dxz.model.model_factory import VisionModelConfig, LanguageModelConfig, VisionModel, LanguageModel, ModelFactory
from dxz.engine.worker import WorkerConfig, WorkerContext, getWorker, Worker
from dxz.memory.virtual_kv_cache import VirtualKVCache
from dxz.memory.memory_management import MemoryManagementUnit, getMemoryManagementUnit
from dxz.engine.scheduler import BatchRequest


@dataclass
class ExecutorConfig:
    use_flash_infer: bool = False
    batch_image_embed_forward: bool = True
    multi_streams_forward: bool = False
    multi_threads_forward: bool = False

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> 'ExecutorConfig':
        attrs = [attr.name for attr in fields(cls)]
        config = cls(
            **{attr: getattr(args, attr) for attr in attrs}
        )
        return config

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--use-flash-infer', action='store_true', help='Enable flash infer attention kernel, default kernel is flash attention')
        parser.add_argument('--batch-image-embed-forward', action='store_true', help='Enable batch image embedding forwarding.')
        parser.add_argument('--multi-streams-forward', action='store_true', help='Enable multi-stream forwarding.')
        parser.add_argument('--multi-threads-forward', action='store_true', help='Enable multi-thread forwarding.')
        parser = WorkerConfig.add_cli_args(parser)
        return parser


@dataclass
class ExecutorContext:
    model_factory_config: ModelFactoryConfig
    block_size: int
    mmu: MemoryManagementUnit
    worker: Worker = None


class Executor:
    def execute(self, contexts: BatchRequest):
        raise NotImplementedError


class BatchFillExecutor(Executor):
    def __init__(self, config: ExecutorConfig, context: ExecutorContext):
        model_factory = getModelFactory(context.model_factory_config, ModelFactoryContext(process_group=None))
        self.worker = context.worker
        self.vision_model_config = model_factory.getVisionModelConfig()
        self.language_model_config = model_factory.getLanguageModelConfig()

        self.config = config
        self.context = context
        self.dtype = context.model_factory_config.dtype
        self.device = context.model_factory_config.device
        self.mmu = context.mmu

        self.workspace_buffer = None
        self.batch_prefill_with_paged_kvcache_wrapper = None
        self.batch_decode_with_paged_kvcache_wrapper = None
        if config.use_flash_infer:
            import flashinfer
            self.workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
            self.batch_prefill_with_paged_kvcache_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.workspace_buffer, "NHD")
            self.batch_decode_with_paged_kvcache_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(self.workspace_buffer, "NHD")

    def execute_image_embed(self, contexts: BatchRequest) -> Tensor:
        if len(contexts) == 0:
            return None
        pixel_values: list[Tensor] = []
        for rcb, inst in contexts:
            pixel_values.append(inst.pixel_values)
            inst.pixel_values = None
        pixel_values = torch.cat(pixel_values, dim=0).to(dtype=self.dtype, device=self.device)
        image_features = self.worker.execute_vision_model(pixel_values, VisionModelParameters(return_last_layer_attention=False)).image_features
        return image_features
            

    def execute(self, contexts: BatchRequest):
        if len(contexts) == 0:
            return {}

        # 1. allocate memory if necessary
        for rcb, _ in contexts:
            if len(rcb.virtual_kv_caches) == 0:
                rcb.virtual_kv_caches = self.mmu.allocate_virtual_kv_caches(rcb.n_virtual_kv_caches)

        # 2. filter out request which need image embed
        batch_image_fill = BatchRequest()
        list_image_features: list[Tensor] = []
        for seq, inst in contexts:
            if isinstance(inst, ImageFill):
                batch_image_fill.append(seq)
            elif isinstance(inst, ImageEmbedFill):
                list_image_features.append(inst.image_features)
                inst.image_features = None
        if len(batch_image_fill) > 0 and len(list_image_features) > 0:
            raise Exception('not support pixel value and image embed batch')

        if len(list_image_features) == 0:
            image_features = self.execute_image_embed(batch_image_fill)
        else:
            image_features = torch.cat(list_image_features, dim=0).to(dtype=self.dtype, device=self.device)

        token_ids         : list[int] = []
        position_ids      : list[int] = []
        selected_token_ids: list[int] = []
        attention_params_builders = [AttentionParametersBuilder(
            num_qo_heads = self.language_model_config.n_qo_heads,
            num_kv_heads = self.language_model_config.n_kv_heads,
            head_dim = self.language_model_config.head_dim, 
            block_size = self.context.block_size, 
            device = self.device, 
            flash_infer_batch_prefill_handler = self.batch_prefill_with_paged_kvcache_wrapper if layer_id == 0 else None, 
            flash_infer_batch_decode_handler = self.batch_decode_with_paged_kvcache_wrapper if layer_id == 0 else None, 
        ) for layer_id in range(self.language_model_config.n_layers)]
        for seq, inst in contexts:
            token_ids += inst.token_ids
            position_ids += inst.position_ids
            if inst.sample:
                selected_token_ids.append(len(token_ids) - 1)

            for layer_id in range(self.language_model_config.n_layers):
                virtual_kv_cache = seq.virtual_kv_caches[inst.kv_cache_ids[layer_id]]
                slot_ids = self.mmu.set(virtual_kv_cache, inst.cache_ids[layer_id])
                attention_params_builders[layer_id].add_request(
                    q_seq_len = len(inst.token_ids), 
                    kv_seq_len = virtual_kv_cache.n_kv_cache_tokens, 
                    new_cache_slots = slot_ids, 
                    block_table = virtual_kv_cache.block_table
                )

        layers_attention_params: list[AttentionParameters] = []
        for layer_id in range(self.language_model_config.n_layers):
            attention_params_builder = attention_params_builders[layer_id]
            attention_params_builder.add_kv_cache(self.mmu.get_layer_kv_cache(layer_id))
            layers_attention_params.append(attention_params_builder.build_attention_parameters()[0])

        model_params = LanguageModelParameters(
            attention_params = layers_attention_params, 
            all_sequences_decode = layers_attention_params[0].all_sequences_decode, 
            selected_token_ids = selected_token_ids
        )
        input_ids_tensor = torch.tensor(token_ids, dtype=torch.int, device=self.device)
        position_ids_tensor = torch.tensor(position_ids, dtype=torch.int, device=self.device)

        sample_token_ids = self.worker.execute_language_model(input_ids_tensor, image_features, position_ids_tensor, model_params).sample_token_ids
        sample_token_ids = sample_token_ids.tolist()

        if len(selected_token_ids) > 0:
            t = time.perf_counter()
            i = 0
            for rcb, instruction in contexts:
                if (isinstance(instruction, Fill)) and instruction.sample:
                    next_token_id = sample_token_ids[i]
                    is_last_token =  instruction.sample_dst is None
                    if not is_last_token:
                        instruction.sample_dst.token_ids = [next_token_id]
                    rcb.metric.tokens_time.append(t)
                    for output_token_processor in rcb.output_token_processors:
                        output_token_processor.append_token_id(next_token_id, is_last_token)
                    i += 1

        for rcb, _ in contexts:
            rcb.instructions.curr = rcb.instructions.curr.next

class ImageEmbedExecutor(Executor):
    def __init__(self, context: ExecutorContext):
        self.context = context
        self.worker = context.worker

    def execute(self, contexts: BatchRequest):
        if len(contexts) == 0:
            return
        for rcb, instruction in contexts:
            pixel_values = instruction.pixel_values.to(self.context.dtype).to(self.context.device)
            vision_params = VisionModelParameters(return_last_layer_attention=False)
            image_features = self.worker.execute_vision_model(pixel_values, vision_params).image_features
            instruction.image_features_dst.image_features = image_features

        for rcb, _ in contexts:
            rcb.instructions.curr = rcb.instructions.curr.next

        return


class BatchImageEmbedExecutor(Executor):
    def __init__(self, context: ExecutorContext):
        self.context = context
        self.worker = context.worker

    def execute(self, contexts: BatchRequest):
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
        image_features = self.worker.execute_vision_model(pixel_values, vision_params).image_features

        left = 0
        for i, (rcb, instruction) in enumerate(contexts):
            right = left + n_images[i]
            instruction.image_features_dst.image_features = image_features[left: right, :, :]
            left += n_images[i]

        for rcb, _ in contexts:
            rcb.instructions.curr = rcb.instructions.curr.next


class MultiStreamsDecorator(Executor):
    def __init__(self, stream: torch.cuda.Stream, executor: Executor):
        self.stream = stream
        self.executor = executor

    def execute(self, contexts: BatchRequest):
        if len(contexts) == 0:
            return
        with torch.cuda.stream(self.stream):
            self.executor.execute(contexts)
            self.stream.synchronize()


class MultiThreadsDecorator:
    def __init__(self, pool: ThreadPoolExecutor, executor: Executor):
        self.pool = pool
        self.executor = executor

    def execute(self, contexts: BatchRequest):
        if len(contexts) == 0:
            return

        future = self.pool.submit(self.executor.execute, contexts)
        return future


class InstructionExecutor:
    def __init__(self, config: ExecutorConfig, context: ExecutorContext):
        self.config = config
        self.context = context

        if self.config.batch_image_embed_forward:
            self.image_embed_executor = BatchImageEmbedExecutor(context)
        else:
            self.image_embed_executor = ImageEmbedExecutor(context)

        self.fill_executor = BatchFillExecutor(config, context)

        self.streams: list[torch.cuda.Stream] = []
        if self.config.multi_streams_forward:
            stream = torch.cuda.Stream()
            self.streams.append(stream)
            self.image_embed_executor = MultiStreamsDecorator(stream, self.image_embed_executor)

        self.pool: Optional[ThreadPoolExecutor] = None
        if self.config.multi_threads_forward:
            self.pool = ThreadPoolExecutor(max_workers=1)
            self.image_embed_executor = MultiThreadsDecorator(self.pool, self.image_embed_executor)

    def execute_fill(self, contexts: BatchRequest):
        return self.fill_executor.execute(contexts)

    def execute_image_embed(self, contexts: BatchRequest):
        return self.image_embed_executor.execute(contexts)

    def execute_empty(self, contexts: BatchRequest):
        for rcb, _ in contexts:
            rcb.instructions.curr = rcb.instructions.curr.next