from collections import deque
import time
import queue
from queue import Queue
from typing import Literal, Optional, Union
from dataclasses import dataclass, fields
from hydrainfer.engine import Instruction, Fill, TextFill, EmptyInstruction, ImageEmbedFill, ImageEmbed, RequestControlBlock, BatchSchedulerProfiler, BatchRequest, PullCache
from hydrainfer.memory import TokenCacheBlockManager
from hydrainfer.utils.allocate import IncreaingAllocator
import argparse
from hydrainfer.utils.logger import getLogger
logger = getLogger(__name__)


@dataclass
class BatchSchedulerMetrics:
    n_running_requests: int
    n_requests_waiting_migrate: int


@dataclass
class BatchSchedulerConfig:
    priority: Literal['prefill', 'decode'] = 'prefill'
    max_running_requests: int = 15
    chunked_prefill: bool = True
    debug: bool = False


@dataclass
class BatchSchedulerContext:
    profiler: BatchSchedulerProfiler
    kv_cache_block_manager: TokenCacheBlockManager
    image_cache_block_manager: TokenCacheBlockManager


class BatchScheduler:
    def __init__(self, config: BatchSchedulerConfig, context: BatchSchedulerContext):
        self.config = config
        self.profiler = context.profiler
        self.image_budgets = self.profiler.profile_image_budgets()
        self.token_budgets = self.profiler.profile_token_budgets()

        # self.waiting = Queue()
        self.waiting = deque()

        self.running: list[RequestControlBlock] = []
        self.step_cnt = 0
        self.sid_allocator = IncreaingAllocator(first_value=1)

        self.max_overload_requests = config.max_running_requests
        self.running_cnt = 0
        self.migrating_cnt = 0

    def migrating_acquire(self):
        # migrating_acquire and migrating_release are used to count how many request need to be pulled
        # we need this count because avoid sender OOM when there are too many waiting migrate requests and running requests
        assert self.migrating_cnt < self.config.max_running_requests + self.max_overload_requests, f'invalid acquire'
        self.migrating_cnt += 1

    def migrating_release(self):
        assert self.migrating_cnt > 0, f'invalid release'
        self.migrating_cnt -= 1

    def _stamp_queuing_begin_time(self, rcb: RequestControlBlock):
        if isinstance(rcb.current_instruction(), ImageEmbed):
            rcb.metric.encode_queueing.append(time.perf_counter())
            return
        if len(rcb.metric.prefill_queueing) == 0:
            rcb.metric.prefill_queueing.append(time.perf_counter())
            return
        if len(rcb.metric.decode_queueing) == 0:
            rcb.metric.decode_queueing.append(time.perf_counter())
            return

    def _stamp_queuing_end_time(self, rcb: RequestControlBlock):
        if len(rcb.metric.encode_queueing) == 1:
            rcb.metric.encode_queueing.append(time.perf_counter())
            return
        if len(rcb.metric.prefill_queueing) == 1:
            rcb.metric.prefill_queueing.append(time.perf_counter())
            return
        if len(rcb.metric.decode_queueing) == 1:
            rcb.metric.decode_queueing.append(time.perf_counter())
            return

    def schedule_new(self, rcb: RequestControlBlock):
        rcb.sid = self.sid_allocator.allocate()
        if isinstance(rcb.current_instruction(), PullCache):
            self.waiting.appendleft(rcb)
        else:
            self.waiting.append(rcb)

        self._stamp_queuing_begin_time(rcb)

    def schedule_running(self, rcb: RequestControlBlock):
        self.running.append(rcb)
        self._stamp_queuing_end_time(rcb)

    def step(self) -> BatchRequest:
        self.step_cnt += 1
        schedule_time = time.perf_counter()
        # 1. get enough requests to participate in the batch
        while len(self.running) < self.config.max_running_requests - self.migrating_cnt and len(self.waiting) > 0:
            rcb = self.waiting.popleft()
            self.schedule_running(rcb)
        # we are risking dead pull cache lock when ED Node' all requests are waiting P Node pull and P Node's all requests are waiting ED Node pull cache
        # so we need to limit the number of new requests
        while len(self.running) < self.config.max_running_requests - self.migrating_cnt + self.max_overload_requests and len(self.waiting) > 0 and isinstance(self.waiting[0].current_instruction(), PullCache):
            rcb = self.waiting.popleft()
            self.schedule_running(rcb)

        self.running_cnt = len(self.running)
        if len(self.running) == 0:
            return []

        batch_fill_tokens = 0
        batch_embed_images = 0
        embed_seqs  : list[RequestControlBlock] = []
        prefill_seqs: list[RequestControlBlock] = []
        decode_seqs : list[RequestControlBlock] = []
        this_step: list[RequestControlBlock] = []
        next_step: list[RequestControlBlock] = []
        # 1. classify seqs
        for seq in self.running:
            inst = seq.instructions.curr
            if isinstance(inst, Fill):
                if len(inst.token_ids) == 1:
                    decode_seqs.append(seq)
                else:
                    prefill_seqs.append(seq)
            elif isinstance(inst, ImageEmbed):
                embed_seqs.append(seq)
            else:
                this_step.append(seq)

        # 2. batch image embed
        for seq in embed_seqs:
            if batch_embed_images < self.image_budgets:
                this_step.append(seq)
                batch_embed_images += 1 # todo cope with multi image
            else:
                next_step.append(seq)

        # 3. batch prefill and decode
        fill_seqs = prefill_seqs + decode_seqs if self.config.priority == 'prefill' else decode_seqs + prefill_seqs
            
        for seq in fill_seqs:
            inst = seq.instructions.curr
            n_tokens = len(inst.token_ids)
            if batch_fill_tokens + n_tokens <= self.token_budgets:
                this_step.append(seq)
                batch_fill_tokens += n_tokens
            elif batch_fill_tokens < self.token_budgets and n_tokens > 1 and self.config.chunked_prefill and (isinstance(inst, TextFill) or isinstance(inst, ImageEmbedFill)): # if it is prefill and we can chunk part of it
                chunk_size = self.token_budgets - batch_fill_tokens
                inst.chunk_prefill(chunk_size)
                this_step.append(seq)
                batch_fill_tokens += chunk_size
            elif batch_fill_tokens == 0: # avoid the prefill larger than token_budgets block
                this_step.append(seq)
                batch_fill_tokens += n_tokens
            else:
                next_step.append(seq)

        if self.config.debug:
            logger.debug(f'------------------------------ scheduler step {self.step_cnt} ------------------------------')
            logger.debug(f'sid : ' + ' '.join(f'{seq.sid: 2}'                 for seq in this_step))
            logger.debug(f'inst: ' + ' '.join(f'{seq.instructions.curr}' for seq in this_step))
            logger.debug(f'batch images {batch_embed_images}')
            logger.debug(f'batch tokens {batch_fill_tokens}')

        self.running = next_step
        return BatchRequest(this_step)

    def get_metrics(self) -> BatchSchedulerMetrics:
        return BatchSchedulerMetrics(
            n_running_requests=self.running_cnt, 
            n_requests_waiting_migrate = self.migrating_cnt, 
        )
