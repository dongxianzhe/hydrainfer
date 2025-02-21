import zmq
import time
import torch
import argparse
import concurrent
from torch import Tensor
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields, field
from dxz.engine import RequestControlBlock, Instruction, ImageFill, ImageEmbedFill, Fill, EmptyInstruction
from dxz.engine import PrintTextOutputTokenProcessor
from dxz.layer.causal_attention import AttentionParametersBuilder, AttentionParameters
from dxz.model.parameters import LanguageModelParameters, VisionModelParameters
from dxz.model.model_factory import ModelFactory, getModelFactory, ModelFactoryConfig, ModelFactoryContext
from dxz.model.model_factory import VisionModelConfig, LanguageModelConfig, VisionModel, LanguageModel, ModelFactory
from dxz.engine.worker import WorkerConfig, WorkerContext, getWorker, Worker
from dxz.memory import TokenCacheBlockManager, KVCache
from dxz.engine.scheduler import BatchRequest
from dxz.utils.config_util import CLIConfig
from dxz.request import OfflineInferenceOutput


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
class ExecutorConfig(CLIConfig):
    use_flash_infer: bool = False
    multi_streams_forward: bool = False
    multi_threads_forward: bool = False
    model_factory_config: ModelFactoryConfig = field(default_factory=ModelFactoryConfig)

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser, prefix: str="--") -> argparse.ArgumentParser:
        parser.add_argument(f'{prefix}use-flash-infer', action='store_true', help='Enable flash infer attention kernel, default kernel is flash attention')
        parser.add_argument(f'{prefix}multi-streams-forward', action='store_true', help='Enable multi-stream forwarding.')
        parser.add_argument(f'{prefix}multi-threads-forward', action='store_true', help='Enable multi-thread forwarding.')
        cls.add_sub_configs_cli_args(cls, parser, prefix)
        return parser


@dataclass
class ExecutorContext:
    kv_cache_block_manager: TokenCacheBlockManager
    image_cache_block_manager: TokenCacheBlockManager
    worker: Worker = None
    zmq_send: Optional[zmq.sugar.socket.Socket] = None


class Executor:
    def execute(self, contexts: BatchRequest):
        raise NotImplementedError


class BatchFillExecutor(Executor):
    def __init__(self, config: ExecutorConfig, context: ExecutorContext):
        self.config = config
        self.context = context

        model_factory = getModelFactory(config.model_factory_config, ModelFactoryContext(process_group=None))
        self.worker = context.worker
        self.vision_model_config = model_factory.getVisionModelConfig()
        self.language_model_config = model_factory.getLanguageModelConfig()
        self.tokenizer = model_factory.getTokenizer()

        self.dtype = config.model_factory_config.dtype
        self.device = config.model_factory_config.device
        self.block_mangaer = context.kv_cache_block_manager
        self.image_block_manager = context.image_cache_block_manager

        self.workspace_buffer = None
        self.batch_prefill_with_paged_kvcache_wrapper = None
        self.batch_decode_with_paged_kvcache_wrapper = None
        if config.use_flash_infer:
            import flashinfer
            self.workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
            self.batch_prefill_with_paged_kvcache_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.workspace_buffer, "NHD")
            self.batch_decode_with_paged_kvcache_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(self.workspace_buffer, "NHD")
        self.print_text_output_token_processor = PrintTextOutputTokenProcessor(self.tokenizer)

    def _execute_image_embed(self, contexts: BatchRequest) -> Tensor:
        if len(contexts) == 0:
            return None
        pixel_values: list[Tensor] = []
        for rcb, inst in contexts:
            pixel_values.append(inst.pixel_values)
            inst.pixel_values = None
        pixel_values = torch.cat(pixel_values, dim=0).to(dtype=self.dtype, device=self.device)
        image_features = self.worker.execute_vision_model(pixel_values, VisionModelParameters(return_last_layer_attention=False)).image_features
        return image_features
            
    def execute(self, contexts: BatchRequest) -> Future:
        if len(contexts) == 0:
            return EmptyFuture()

        # 1. allocate memory if necessary
        for rcb, _ in contexts:
            if rcb.virtual_kv_cache is None:
                rcb.virtual_kv_cache = self.block_mangaer.allocate_virtual_cache()

        # 2. filter out request which need image embed
        batch_image_fill = BatchRequest()
        image_tokens: list[Tensor] = []
        for rcb, inst in contexts:
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
        for rcb, inst in contexts:
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

        if len(selected_token_ids) > 0:
            t = time.perf_counter()
            i = 0
            for rcb, instruction in contexts:
                if (isinstance(instruction, Fill)) and instruction.sample:
                    next_token_id = sample_token_ids[i]
                    is_last_token =  instruction.sample_dst is None
                    if not is_last_token:
                        instruction.sample_dst.token_ids = [next_token_id]
                    rcb.metric.token_times.append(t)
                    # process output tokens
                    rcb.output_token_ids.append(next_token_id)
                    for output_token_processor in rcb.output_token_processors:
                        output_token_processor.append_token_id(next_token_id, is_last_token)
                    if rcb.output_token_params.print_output_text:
                        self.print_text_output_token_processor.append_token_id(next_token_id, is_last_token)
                    if self.context.zmq_send:
                        if rcb.output_token_params.is_offline_output:
                            if rcb.output_token_params.is_stream_output:
                                raise Exception('offline inference is not support stream output')
                            else:
                                if is_last_token:
                                    self.context.zmq_send.send_pyobj(OfflineInferenceOutput(
                                        text = self.tokenizer.decode(rcb.output_token_ids),
                                        output_token_ids = rcb.output_token_ids, 
                                        arrival_time  = rcb.metric.arrival_time, 
                                        finished_time = rcb.metric.finished_time, 
                                        token_times = rcb.metric.token_times,

                                        ttft = rcb.metric.token_times[0] - rcb.metric.arrival_time, 
                                        tpot = [
                                            rcb.metric.token_times[i] - rcb.metric.token_times[i-1]
                                            for i in range(1, len(rcb.metric.token_times))
                                        ], 
                                    ))
                        else:
                            if rcb.output_token_params.is_stream_output:
                                self.context.zmq_send.send_pyobj((rcb.request_id, [next_token_id]))
                            else:
                                if is_last_token:
                                    self.context.zmq_send.send_pyobj((rcb.request_id, rcb.output_token_ids))

                    i += 1

        for rcb, _ in contexts:
            rcb.instructions.curr = rcb.instructions.curr.next
        
        return EmptyFuture()


class BatchImageEmbedExecutor(Executor):
    def __init__(self, config: ExecutorConfig, context: ExecutorContext):
        self.config = config
        self.context = context
        self.worker = context.worker
        self.block_manager = context.image_cache_block_manager
        model_factory = getModelFactory(config.model_factory_config, ModelFactoryContext(process_group=None))
        self.language_model_config = model_factory.getLanguageModelConfig()
        self.n_qo_heads = self.language_model_config.n_qo_heads
        self.head_dim = self.language_model_config.head_dim
        self.device = config.model_factory_config.device

    def execute(self, contexts: BatchRequest) -> Future:
        if len(contexts) == 0:
            return EmptyFuture()
        for rcb, _ in contexts:
            if rcb.virtual_image_cache is None:
                rcb.virtual_image_cache = self.block_manager.allocate_virtual_cache()

        new_cache_slots: list[int] = []
        batch_pixel_values: list[Tensor] = []
        for rcb, inst in contexts:
            pixel_values = inst.pixel_values.to(self.config.model_factory_config.dtype).to(self.config.model_factory_config.device) # (n_images, n_channels, width, height)
            batch_pixel_values.append(pixel_values)
            inst.pixel_values = None
            slot_ids = self.block_manager.set(rcb.virtual_image_cache, inst.cache_ids) 
            new_cache_slots += slot_ids 
        pixel_values = torch.cat(batch_pixel_values, dim=0) # (total_images, n_channels, width, height)

        vision_params = VisionModelParameters(return_last_layer_attention=False)
        image_features = self.worker.execute_vision_model(pixel_values, vision_params).image_features
        # (total_images, num_image_tokens, hidden_size)
        image_tokens = image_features.reshape(-1, self.n_qo_heads, self.head_dim) # (total_image_tokens, n_qo_heads, head_size)
        new_cache_slots = torch.tensor(new_cache_slots, dtype=torch.int, device=self.device)
        token_cache = self.block_manager.get_layer_cache(layer_id=0)
        token_cache.set_caches(new_cache_slots, [image_tokens])

        for rcb, _ in contexts:
            rcb.instructions.curr = rcb.instructions.curr.next

        return EmptyFuture()


class MultiStreamsDecorator(Executor):
    def __init__(self, stream: torch.cuda.Stream, executor: Executor):
        self.stream = stream
        self.executor = executor

    def execute(self, contexts: BatchRequest) -> Future:
        if len(contexts) == 0:
            return EmptyFuture()

        with torch.cuda.stream(self.stream):
            self.executor.execute(contexts)
            self.stream.synchronize()
        return EmptyFuture()


class MultiThreadsDecorator:
    def __init__(self, pool: ThreadPoolExecutor, executor: Executor):
        self.pool = pool
        self.executor = executor

    def execute(self, contexts: BatchRequest) -> Future:
        if len(contexts) == 0:
            return EmptyFuture()

        future = self.pool.submit(self.executor.execute, contexts)

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

    def execute_fill(self, contexts: BatchRequest) -> Future:
        return self.fill_executor.execute(contexts)

    def execute_image_embed(self, contexts: BatchRequest) -> Future:
        return self.image_embed_executor.execute(contexts)

    def execute_empty(self, contexts: BatchRequest) -> Future:
        if len(contexts) == 0:
            return EmptyFuture()
        for rcb, _ in contexts:
            rcb.instructions.curr = rcb.instructions.curr.next
        return EmptyFuture()