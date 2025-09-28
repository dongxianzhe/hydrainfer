import zmq
import time
import torch
import argparse
import concurrent
from torch import Tensor
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields, field
from hydrainfer.engine import RequestControlBlock, Instruction, ImageFill, ImageEmbedFill, Fill, EmptyInstruction, ImageEmbed
from hydrainfer.engine import PrintTextOutputTokenProcessor
from hydrainfer.layer.causal_attention import AttentionParametersBuilder, AttentionParameters
from hydrainfer.model.parameters import LanguageModelParameters, VisionModelParameters
from hydrainfer.model.model_factory import ModelFactory, getModelFactory, ModelFactoryConfig, ModelFactoryContext
from hydrainfer.model.model_factory import VisionModelConfig, LanguageModelConfig, VisionModel, LanguageModel, ModelFactory
from hydrainfer.engine.worker import WorkerConfig, WorkerContext, getWorker, Worker
from hydrainfer.memory import TokenCacheBlockManager, KVCache
from hydrainfer.engine import BatchRequest
from hydrainfer.request import OfflineInferenceOutput
from hydrainfer.utils.torch_utils import str2dtype, str2device
from hydrainfer.utils.logger import getLogger
logger = getLogger(__name__)

try:
    import flashinfer
except ImportError:
    logger.warning('flashinfer import failed')
    flashinfer = None


class Future:
    def get(self):
        raise NotImplementedError


class EmptyFuture(Future):
    def get(self):
        pass


class ThreadPoolExecutorFuture(Future):
    def __init__(self, future: concurrent.futures._base.Future):
        self.future = future

    def get(self):
        self.future.result()


class ComposeFuture(Future):
    def __init__(self, futures: list[Future]):
        self.futures = futures

    def get(self):
        for future in self.futures:
            future.get()

@dataclass
class ExecutorConfig:
    use_flash_infer: bool = False
    multi_streams_forward: bool = False
    multi_threads_forward: bool = False
    model: ModelFactoryConfig = field(default_factory=ModelFactoryConfig)


@dataclass
class ExecutorContext:
    kv_cache_block_manager: TokenCacheBlockManager
    image_cache_block_manager: TokenCacheBlockManager
    worker: Worker = None
    zmq_send: Optional[zmq.sugar.socket.Socket] = None


class Executor:
    def execute(self, batch: BatchRequest):
        raise NotImplementedError


class BatchFillExecutor(Executor):
    def __init__(self, config: ExecutorConfig, context: ExecutorContext):
        self.config = config
        self.context = context

        model_factory = getModelFactory(config.model, ModelFactoryContext(process_group=None))
        self.worker = context.worker
        self.vision_model_config = model_factory.getVisionModelConfig()
        self.language_model_config = model_factory.getLanguageModelConfig()
        self.tokenizer = model_factory.getTokenizer()

        self.dtype = str2dtype(config.model.dtype)
        self.device = str2device(config.model.device)
        self.block_mangaer = context.kv_cache_block_manager
        self.image_block_manager = context.image_cache_block_manager

        self.workspace_buffer = None
        self.batch_prefill_with_paged_kvcache_wrapper = None
        self.batch_decode_with_paged_kvcache_wrapper = None
        if config.use_flash_infer and flashinfer is not None:
            self.workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
            self.batch_prefill_with_paged_kvcache_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.workspace_buffer, "NHD")
            self.batch_decode_with_paged_kvcache_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(self.workspace_buffer, "NHD", use_tensor_cores=True)
        self.print_text_output_token_processor = PrintTextOutputTokenProcessor(self.tokenizer)

    def _execute_image_embed(self, batch: BatchRequest) -> Tensor:
        if len(batch) == 0:
            return None
        pixel_values: list[Tensor] = []
        for rcb, inst in batch:
            pixel_values.append(inst.pixel_values)
            inst.pixel_values = None
        pixel_values = torch.cat(pixel_values, dim=0).to(dtype=self.dtype, device=self.device)
        image_features = self.worker.execute_vision_model(pixel_values, VisionModelParameters(return_last_layer_attention=False)).image_features
        return image_features
            
    def execute(self, batch: BatchRequest) -> Future:
        if len(batch) == 0:
            return EmptyFuture()

        # 1. allocate memory if necessary
        for rcb, _ in batch:
            if rcb.virtual_kv_cache is None:
                rcb.virtual_kv_cache = self.block_mangaer.allocate_virtual_cache()

        # 2. filter out request which need image embed
        batch_image_fill = BatchRequest()
        image_tokens: list[Tensor] = []
        for rcb, inst in batch:
            if isinstance(inst, ImageFill):
                batch_image_fill.append(rcb)
            elif isinstance(inst, ImageEmbedFill):
                token_cache = self.image_block_manager.get_layer_cache(layer_id=0)
                slot_ids = self.image_block_manager.v2p(rcb.virtual_image_cache, inst.image_token_cache_ids)
                image_token_cache = token_cache.get_caches()[0]
                image_token_cache = image_token_cache.view(-1, self.language_model_config.n_qo_heads * self.language_model_config.head_dim)
                slot_ids = torch.tensor(slot_ids, dtype=torch.int, device=self.device)
                request_image_tokens = image_token_cache[slot_ids, :]
                image_tokens.append(request_image_tokens)
        if len(batch_image_fill) > 0 and len(image_tokens) > 0:
            raise Exception('not support pixel value and image embed batch')

        if len(image_tokens) == 0:
            image_features = self._execute_image_embed(batch_image_fill)
        else:
            image_features = torch.cat(image_tokens, dim=0).to(dtype=self.dtype, device=self.device)

        token_ids         : list[int] = []
        position_ids      : list[int] = []
        selected_token_ids: list[int] = []
        attention_params_builder = AttentionParametersBuilder(
            num_qo_heads = self.language_model_config.n_qo_heads,
            num_kv_heads = self.language_model_config.n_kv_heads,
            head_dim = self.language_model_config.head_dim, 
            block_size = self.context.kv_cache_block_manager.config.block_size, 
            device = self.device, 
            flash_infer_batch_prefill_handler = self.batch_prefill_with_paged_kvcache_wrapper, 
            flash_infer_batch_decode_handler = self.batch_decode_with_paged_kvcache_wrapper, 
        )
        for rcb, inst in batch:
            token_ids += inst.token_ids
            position_ids += inst.position_ids
            if inst.sample:
                selected_token_ids.append(len(token_ids) - 1)

            virtual_kv_cache = rcb.virtual_kv_cache
            slot_ids = self.block_mangaer.set(virtual_kv_cache, inst.cache_ids)
            attention_params_builder.add_request(
                q_seq_len = len(inst.token_ids), 
                kv_seq_len = virtual_kv_cache.n_cache_tokens, 
                new_cache_slots = slot_ids, 
                block_table = virtual_kv_cache.block_table
            )
        for layer_id in range(self.language_model_config.n_layers):
            attention_params_builder.add_kv_cache(KVCache.from_token_cache(self.block_mangaer.get_layer_cache(layer_id)))
        layers_attention_params = attention_params_builder.build_attention_parameters()
        model_params = LanguageModelParameters(
            attention_params = layers_attention_params, 
            all_sequences_decode = layers_attention_params[0].all_sequences_decode, 
            selected_token_ids = selected_token_ids
        )
        input_ids_tensor = torch.tensor(token_ids, dtype=torch.int, device=self.device)
        position_ids_tensor = torch.tensor(position_ids, dtype=torch.int, device=self.device)

        sample_token_ids = self.worker.execute_language_model(input_ids_tensor, image_features, position_ids_tensor, model_params).sample_token_ids
        sample_token_ids = sample_token_ids.tolist()

        if len(selected_token_ids) == 0:
            batch.step()
            return EmptyFuture()

        token_time = time.perf_counter()
        i = 0
        for rcb, inst in batch:
            if not isinstance(inst, Fill) or not inst.sample:
                continue

            next_token_id = sample_token_ids[i]
            i += 1

            if not inst.is_chunked:
                rcb.metric.token_times.append(token_time)
                rcb.output_token_ids.append(next_token_id)

            if inst.sample_dst:
                inst.sample_dst.token_ids = [next_token_id]

            is_last_token: bool = rcb.is_finished()

            # process output tokens
            if not inst.is_chunked:
                for output_token_processor in rcb.output_token_processors:
                    output_token_processor.append_token_id(next_token_id, is_last_token)
                if rcb.output_token_params.print_output_text:
                    self.print_text_output_token_processor.append_token_id(next_token_id, is_last_token)

            if not self.context.zmq_send or not rcb.output_token_params.zmq_output:
                continue

            if inst.is_chunked:
                continue

            content = self.tokenizer.decode(next_token_id)
            self.context.zmq_send.send_pyobj((rcb.request_id, content))
            if is_last_token:
                self.context.zmq_send.send_pyobj((rcb.request_id, None))

        batch.step()
        return EmptyFuture()


class BatchImageEmbedExecutor(Executor):
    def __init__(self, config: ExecutorConfig, context: ExecutorContext):
        self.config = config
        self.context = context
        self.worker = context.worker
        self.block_manager = context.image_cache_block_manager
        model_factory = getModelFactory(config.model, ModelFactoryContext(process_group=None))
        self.language_model_config = model_factory.getLanguageModelConfig()
        self.n_qo_heads = self.language_model_config.n_qo_heads
        self.head_dim = self.language_model_config.head_dim
        self.dtype = str2dtype(config.model.dtype)
        self.device = str2device(config.model.device)

    def execute(self, batch: BatchRequest) -> Future:
        if len(batch) == 0:
            return EmptyFuture()
        for rcb, _ in batch:
            if rcb.virtual_image_cache is None:
                rcb.virtual_image_cache = self.block_manager.allocate_virtual_cache()

        new_cache_slots: list[int] = []
        images_size: list[tuple[int, int]] = []
        batch_pixel_values: list[Tensor] = []
        for rcb, inst in batch:
            pixel_values = inst.pixel_values.to(self.dtype).to(self.device) # (n_images, n_channels, width, height)
            batch_pixel_values.append(pixel_values)
            inst.pixel_values = None
            slot_ids = self.block_manager.set(rcb.virtual_image_cache, inst.cache_ids) 
            new_cache_slots += slot_ids 
            images_size += inst.images_size

        vision_params = VisionModelParameters(return_last_layer_attention=False, original_image_sizes=images_size)
        image_features = self.worker.execute_vision_model(batch_pixel_values, vision_params).image_features
        # (total_images, num_image_tokens, hidden_size)
        image_tokens = image_features.reshape(-1, self.n_qo_heads, self.head_dim) # (total_image_tokens, n_qo_heads, head_size)
        new_cache_slots = torch.tensor(new_cache_slots, dtype=torch.int, device=self.device)
        token_cache = self.block_manager.get_layer_cache(layer_id=0)
        token_cache.set_caches(new_cache_slots, [image_tokens])

        batch.step()

        return EmptyFuture()


class MultiStreamsDecorator(Executor):
    def __init__(self, stream: torch.cuda.Stream, executor: Executor):
        self.stream = stream
        self.executor = executor

    def execute(self, batch: BatchRequest) -> Future:
        if len(batch) == 0:
            return EmptyFuture()

        with torch.cuda.stream(self.stream):
            self.executor.execute(batch)
            self.stream.synchronize()
        return EmptyFuture()


class MultiThreadsDecorator:
    def __init__(self, pool: ThreadPoolExecutor, executor: Executor):
        self.pool = pool
        self.executor = executor

    def execute(self, batch: BatchRequest) -> Future:
        if len(batch) == 0:
            return EmptyFuture()

        future = self.pool.submit(self.executor.execute, batch)

        return ThreadPoolExecutorFuture(future)


class InstructionExecutor:
    def __init__(self, config: ExecutorConfig, context: ExecutorContext):
        self.config = config
        self.context = context

        self.image_embed_executor = BatchImageEmbedExecutor(config, context)

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

    def execute_fill(self, batch: BatchRequest) -> Future:
        future = self.fill_executor.execute(batch)

        return future

    def execute_image_embed(self, batch: BatchRequest) -> Future:
        future = self.image_embed_executor.execute(batch)
        return future

    def execute_empty(self, batch: BatchRequest) -> Future:
        if len(batch) == 0:
            return EmptyFuture()
        batch.step()
        return EmptyFuture()