import copy
import random
import time
import torch
import functools
import numpy as np
from PIL import Image
from typing import Callable, Optional
from dataclasses import dataclass, field
from hydrainfer.engine import BatchRequest, InstructionExecutor, InstructionListBuilder, ImageEmbed, RequestControlBlock, TextFill, Instruction, Future
from hydrainfer.request import SamplingParameters
from hydrainfer.memory import TokenCacheBlockManager
from hydrainfer.engine.output_token_processor import OutputTokenParams
from hydrainfer.model import ModelFactoryConfig, ModelFactoryContext, getModelFactory
from hydrainfer.utils.torch_utils import str2device, str2dtype
from hydrainfer.utils.logger import getLogger
logger = getLogger(__name__)

@dataclass
class BatchSchedulerProfilerConfig:
    model: ModelFactoryConfig = field(default_factory=ModelFactoryConfig)
    tpot_slo: float = 0.4
    debug: bool = False


@dataclass
class BatchSchedulerProfilerContext:
    executor: InstructionExecutor
    kv_cache_block_manager: TokenCacheBlockManager
    image_cache_block_manager: TokenCacheBlockManager


class BatchSchedulerProfiler:
    def __init__(self, config: BatchSchedulerProfilerConfig, context: BatchSchedulerProfilerContext):
        self.config = config
        self.context = context
        model_factory = getModelFactory(config=config.model, context=ModelFactoryContext())
        self.vision_config = model_factory.getVisionModelConfig()
        self.language_confeig = model_factory.getLanguageModelConfig()
        self.processor = model_factory.getProcessor()

        self.dtype = str2dtype(config.model.dtype)
        self.device = str2device(config.model.device)

        self.executor = context.executor

        self.n_warmup_iter = 3
        self.n_profile_iter = 3


    def _prepare_rcb(self, inst: Instruction) -> RequestControlBlock:
        inst_builder = InstructionListBuilder()
        for i in range(self.n_warmup_iter + self.n_profile_iter):
            inst_builder.append(copy.deepcopy(inst))
        rcb = RequestControlBlock()
        rcb.request_id=random.randint(1, 9999)
        rcb.instructions=inst_builder.build_instruction_list()
        rcb.output_token_params=OutputTokenParams(zmq_output=False)
        rcb.sampling_params=SamplingParameters()
        return rcb
    
    def _prepare_encode_batch(self, batch_size: int) -> BatchRequest:
        height, width, n_channel = 336, 336, 3
        random_array = np.random.randint(0, 256, (height, width, n_channel), dtype=np.uint8)
        image = Image.fromarray(random_array)
        images_tensor = self.processor(
            text="", 
            images = image, 
            return_tensors="pt"
        )['pixel_values']
        n_image_tokens = self.vision_config.image_token_caculator.get_num_image_tokens((height, width)) 
        images_tensor = images_tensor.to(self.dtype).to(self.device)
        encode_inst = ImageEmbed(
            pixel_values = images_tensor, 
            cache_ids = list(range(0, n_image_tokens)), 
            images_size=[(height, width)] # original image size
        )

        batch = BatchRequest()
        for _ in range(batch_size):
            batch.append(self._prepare_rcb(encode_inst))
        return batch

    def _prepare_prefill_batch(self, batch_size: int) -> BatchRequest:
        n_prompt_tokens_per_requests = 16
        prefill_inst = TextFill(
            token_ids = list(range(n_prompt_tokens_per_requests)), 
            position_ids = list(range(n_prompt_tokens_per_requests)), 
            cache_ids = list(range(n_prompt_tokens_per_requests)), 
            sample=True, 
            sample_dst=None
        )
        batch = BatchRequest()
        for _ in range(batch_size // n_prompt_tokens_per_requests):
            batch.append(self._prepare_rcb(prefill_inst))
        return batch

    def _prepare_decode_batch(self, batch_size: int, n_cache: int = 512) -> BatchRequest:
        decode_inst = TextFill(
            token_ids = [1], 
            position_ids = [n_cache], 
            cache_ids = [n_cache], 
            sample=True, 
            sample_dst=None
        )
        batch = BatchRequest()

        shared_virtual_kv_cache = self.context.kv_cache_block_manager.allocate_virtual_cache()
        for _ in range(batch_size):
            rcb = self._prepare_rcb(decode_inst)
            rcb.virtual_kv_cache = shared_virtual_kv_cache # use shared kv cache to avoid OOM
            batch.append(rcb)
        return batch

    def _binary_search_max_batch_size(self, left: int, right: int, criterion: Callable[[int], bool]):
        while left < right:
            mid = (left + right + 1) // 2
            try:
                if criterion(mid):
                    # because latency is not increacing with batchsize strictly, we set left = mid + 1 to avoid dead loop
                    left = mid + 1 
                else:
                    right = mid - 1
            except Exception as e:
                right = mid - 1
                if self.config.debug:
                    logger.debug(e)
        return left

    def _free_cache(self, batch: BatchRequest):
        for rcb, _ in batch:
            if rcb.virtual_image_cache:
                self.context.image_cache_block_manager.realloc(rcb.virtual_image_cache, 0)
            if rcb.virtual_kv_cache:
                self.context.kv_cache_block_manager.realloc(rcb.virtual_kv_cache, 0)

    def _criterion(self, mid: int, prepare_batch: Callable[[int], BatchRequest], execute: Callable[[BatchRequest], Future], name: str = "criterion") -> bool:
        batch = prepare_batch(mid)
        for _ in range(self.n_warmup_iter):
            future = execute(batch)
            future.get()

        total_dur = 0.
        for _ in range(self.n_profile_iter):
            start = time.perf_counter()
            future = execute(batch)
            future.get()
            end = time.perf_counter()
            dur = end - start
            total_dur += dur
        avg_dur = total_dur / self.n_profile_iter
        self._free_cache(batch)
        logger.debug(f'{name} binary search [{mid}] profile_avg_dur: {avg_dur} target_tpot_slo {self.config.tpot_slo}')
        return avg_dur < self.config.tpot_slo - 0.01

    def profile_image_budgets(self) -> int:
        logger.debug(f'start profile_image_budgets')
        image_budgets = self._binary_search_max_batch_size(
            left=1, 
            right=8, 
            criterion=functools.partial(
                self._criterion, 
                prepare_batch=self._prepare_encode_batch, 
                execute=self.executor.execute_image_embed, 
                name='image budgets', 
            )
        )
        logger.info(f'auto set image budgets {image_budgets}')
        return image_budgets

    def profile_token_budgets(self) -> int:
        logger.info(f'start profile_token_budgets')
        token_budgets = self._binary_search_max_batch_size(
            left=1, 
            right=2048, 
            criterion=functools.partial(
                self._criterion, 
                prepare_batch=self._prepare_prefill_batch, 
                execute=self.executor.execute_fill, 
                name='token budgets', 
            )
        )
        logger.info(f'auto set token budgets {token_budgets}')
        return token_budgets

    def interference_analysis(self, mode='ep'):
        assert mode in ['ep', 'ed']
        logger.info(f'start analysis interference {mode}')
        # loop image number
        for n_encode in range(10):
            # loop prefill number
            if mode == 'ep':
                n_tokens_list = range(512, 512 * 8 + 1, 512)
            elif mode == 'ed':
                n_tokens_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
            for n_tokens in n_tokens_list:
                # prepare image batch
                encode_batch = self._prepare_encode_batch(n_encode)
                # prepare prefill batch
                if mode == 'ep':
                    fill_batch = self._prepare_prefill_batch(n_tokens)
                # prepare decode batch
                elif mode == 'ed':
                    fill_batch = self._prepare_decode_batch(n_tokens, n_cache = 512)

                def execute():
                    futures = []
                    futures.append(self.executor.execute_image_embed(encode_batch))
                    futures.append(self.executor.execute_fill(fill_batch))
                    for future in futures:
                        future.get()

                # warm up
                for _ in range(self.n_warmup_iter):
                    execute()

                # forward
                total_dur = 0.
                for _ in range(self.n_profile_iter):
                    start = time.perf_counter()
                    execute()
                    end = time.perf_counter()
                    dur = end - start
                    total_dur += dur
                self._free_cache(encode_batch)
                self._free_cache(fill_batch)
                # caculate time
                avg_dur = total_dur / self.n_profile_iter
                # caculate throughput
                throughput = n_tokens / avg_dur
                logger.info('encode, n_tokens, latency, throughput:', n_encode, n_tokens, avg_dur, throughput)
