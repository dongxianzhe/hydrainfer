import copy
import random
import time
import torch
import argparse
import functools
from typing import Callable
from dataclasses import dataclass, field
from dxz.utils.config_util import CLIConfig
from dxz.engine import BatchRequest, InstructionExecutor, InstructionListBuilder, ImageEmbed, RequestControlBlock, TextFill, Instruction, Future
from dxz.memory import TokenCacheBlockManager
from dxz.engine.output_token_processor import OutputTokenParams
from dxz.model import ModelFactoryConfig, ModelFactoryContext, getModelFactory

@dataclass
class BatchSchedulerProfilerConfig(CLIConfig):
    model_factory_config: ModelFactoryConfig = field(default_factory=ModelFactoryConfig)
    profile_batch_config: bool = False
    tpot_slo: float = 0.4

    def add_curr_config_cli_args(cls, parser: argparse.ArgumentParser, prefix: str="--") -> argparse.ArgumentParser:
        parser.add_argument(f'{prefix}profile-batch-config', action='store_true', help='profile best config with tpot slo for batch policy')
        return parser


@dataclass
class BatchSchedulerProfilerContext:
    executor: InstructionExecutor
    kv_cache_block_manager: TokenCacheBlockManager
    image_cache_block_manager: TokenCacheBlockManager


class BatchSchedulerProfiler:
    def __init__(self, config: BatchSchedulerProfilerConfig, context: BatchSchedulerProfilerContext):
        self.config = config
        self.context = context
        model_factory = getModelFactory(config=config.model_factory_config, context=ModelFactoryContext())
        self.vision_config = model_factory.getVisionModelConfig()
        self.language_confeig = model_factory.getLanguageModelConfig()

        self.executor = context.executor

        self.n_warmup_iter = 3
        self.n_profile_iter = 3


    def _prepare_rcb(self, inst: Instruction) -> RequestControlBlock:
        inst_builder = InstructionListBuilder()
        for i in range(self.n_warmup_iter + self.n_profile_iter):
            inst_builder.append(copy.deepcopy(inst))
        rcb = RequestControlBlock(
            request_id=random.randint(1, 9999), 
            instructions=inst_builder.build_instruction_list(), 
            output_token_params=OutputTokenParams(zmq_output=False), 
            sampling_params=None
        )
        return rcb
    
    def _prepare_encode_batch(self, batch_size: int) -> BatchRequest:
        encode_inst = ImageEmbed(
            pixel_values = torch.randn(size=(1, 3, 336, 336), dtype=self.config.model_factory_config.dtype, device=self.config.model_factory_config.device),
            cache_ids = list(range(0, self.vision_config.num_image_tokens)), 
            token_pruning_params = None,
        )

        batch = BatchRequest()
        for _ in range(batch_size):
            batch.append(self._prepare_rcb(encode_inst))
        return batch

    def _prepare_prefill_batch(self, batch_size: int) -> BatchRequest:
        n_prompt_tokens = 16
        prefill_inst = TextFill(
            token_ids = list(range(n_prompt_tokens)), 
            position_ids = list(range(n_prompt_tokens)), 
            cache_ids = list(range(n_prompt_tokens)), 
            sample=True, 
            sample_dst=None
        )
        batch = BatchRequest()
        for _ in range(batch_size // n_prompt_tokens):
            batch.append(self._prepare_rcb(prefill_inst))
        return batch

    def _binary_search_max_batch_size(self, left: int, right: int, criterion: Callable[[int], bool]):
        while left < right:
            mid = (left + right + 1) // 2
            if criterion(mid):
                # because latency is not increacing with batchsize strictly, we set left = mid + 1 to avoid dead loop
                left = mid + 1 
            else:
                right = mid - 1
        return left

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
        for rcb, _ in batch:
            if rcb.virtual_image_cache:
                self.context.image_cache_block_manager.realloc(rcb.virtual_image_cache, 0)
            if rcb.virtual_kv_cache:
                self.context.kv_cache_block_manager.realloc(rcb.virtual_kv_cache, 0)
        print(f'{name} binary search [{mid}] avg_dur: {avg_dur} tpot_slo {self.config.tpot_slo}')
        return avg_dur < self.config.tpot_slo - 0.01

    def profile_image_budgets(self) -> int:
        print(f'start profile_image_budgets')
        image_budgets = self._binary_search_max_batch_size(
            left=1, 
            right=10, 
            criterion=functools.partial(
                self._criterion, 
                prepare_batch=self._prepare_encode_batch, 
                execute=self.executor.execute_image_embed, 
                name='image budgets', 
            )
        )
        print(f'finish profile_image_budgets {image_budgets}')
        return image_budgets

    def profile_token_budgets(self) -> int:
        print(f'start profile_token_budgets')
        token_budgets = self._binary_search_max_batch_size(
            left=1, 
            right=4096, 
            criterion=functools.partial(
                self._criterion, 
                prepare_batch=self._prepare_prefill_batch, 
                execute=self.executor.execute_fill, 
                name='token budgets', 
            )
        )
        print(f'finish profile_token_budgets {token_budgets}')
        return token_budgets