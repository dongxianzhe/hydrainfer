import threading
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
    n_running: int
    n_new: int
    n_migrating: int
    n_waiting_to_pull: int


@dataclass
class BatchSchedulerConfig:
    priority: Literal['prefill', 'decode'] = 'prefill'
    max_running_requests: int = 15
    chunked_prefill: bool = True
    ttft_slo: float = 4
    debug: bool = False


@dataclass
class BatchSchedulerContext:
    profiler: BatchSchedulerProfiler
    kv_cache_block_manager: TokenCacheBlockManager
    image_cache_block_manager: TokenCacheBlockManager


class BatchScheduler:
    def __init__(self, config: BatchSchedulerConfig, context: BatchSchedulerContext):
        self.config = config
        self.context = context
        self.profiler = context.profiler
        self.image_budgets = self.profiler.profile_image_budgets()
        self.token_budgets = self.profiler.profile_token_budgets()

        self.waiting: queue.Queue[RequestControlBlock] = queue.Queue() # new rquest
        self.running: list[RequestControlBlock] = [] # running request
        self.migrating_requests: dict[int, RequestControlBlock] = {} # waiting to be pulled by dst node
        self.pulling: queue.Queue[RequestControlBlock] = queue.Queue() # need pull cache from src node
        self.mutex = threading.Lock()

        self.step_cnt = 0

        self.max_overload_requests = config.max_running_requests
        self.running_cnt = 0

    def schedule_new(self, rcb: RequestControlBlock):
        self.waiting.put(rcb)

    def schedule_running(self, rcb: RequestControlBlock):
        with self.mutex:
            self.running.append(rcb)

    def schedule_waiting_to_be_pulled(self, rcb: RequestControlBlock):
        assert rcb.request_id not in self.migrating_requests
        self.migrating_requests[rcb.request_id] = rcb

    def schedule_waiting_to_pull(self, rcb: RequestControlBlock):
        self.pulling.put(rcb)

    def free_request(self, rcb: RequestControlBlock):
        if rcb.virtual_kv_cache:
            assert self.context.kv_cache_block_manager is not None
            self.context.kv_cache_block_manager.realloc(rcb.virtual_kv_cache, 0)
        if rcb.virtual_image_cache:
            assert self.context.image_cache_block_manager is not None
            self.context.image_cache_block_manager.realloc(rcb.virtual_image_cache, 0)
        if rcb.request_id in self.migrating_requests:
            del self.migrating_requests[rcb.request_id]
        curr_inst = rcb.current_instruction()
        if isinstance(curr_inst, PullCache) and curr_inst.src_node_actor_handle:
            curr_inst.src_node_actor_handle.free_migrating_request.remote(rcb.request_id)

    def free_migrated_request(self, request_id: int):
        assert request_id in self.migrating_requests
        self.free_request(self.migrating_requests[request_id])

    def step(self) -> BatchRequest:
        self.step_cnt += 1
        schedule_time = time.perf_counter()
        # 1. get enough requests to participate in the batch
        while len(self.running) < self.config.max_running_requests - len(self.migrating_requests) and not self.waiting.empty():
            rcb = self.waiting.get() 
            if schedule_time - rcb.arrival_time > self.config.ttft_slo:
                logger.info(f'abondon request {rcb.request_id} because of overload')
                self.free_request(rcb)
            self.schedule_running(rcb)
        # we are risking dead pull cache lock when ED Node' all requests are waiting P Node pull and P Node's all requests are waiting ED Node pull cache
        # so we need to limit the number of new requests
        while len(self.running) < self.config.max_running_requests - len(self.migrating_requests) + self.max_overload_requests and not self.pulling.empty():
            rcb = self.pulling.get()
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

        # 1. allocate cache and skip some chunked prefill tasks because of prefix cache matching
        after_allocated_cache: list[RequestControlBlock] = []
        for rcb in self.running:
            inst = rcb.current_instruction()
            cache_is_enough: bool = True
            if isinstance(inst, Fill):
                if rcb.virtual_kv_cache is None:
                    rcb.virtual_kv_cache = self.context.kv_cache_block_manager.allocate_virtual_cache(inst.hashes)
                    num_prefix_cache_matched = rcb.virtual_kv_cache.n_cache_tokens
                    if num_prefix_cache_matched > 0:
                        assert num_prefix_cache_matched <= len(inst.token_ids)
                        if num_prefix_cache_matched < len(inst.token_ids):
                            inst.chunk_prefill(chunk_size=num_prefix_cache_matched)
                        rcb.step()
                inst = rcb.current_instruction()
                if isinstance(inst, Fill):
                    if not self.context.kv_cache_block_manager.realloc(rcb.virtual_kv_cache, max(rcb.virtual_kv_cache.n_cache_tokens, max(inst.cache_ids) + 1)):
                        logger.warning(f'kv cache is not enough, abandon request {rcb.request_id} decresaing max_running_requests')
                        self.max_overload_requests -= 1
                        self.config.max_running_requests -= 1
                        cache_is_enough = False
            elif isinstance(inst, ImageEmbed):
                if rcb.virtual_image_cache is None:
                    rcb.virtual_image_cache = self.context.image_cache_block_manager.allocate_virtual_cache()
                if not self.context.image_cache_block_manager.realloc(rcb.virtual_image_cache, max(rcb.virtual_image_cache.n_cache_tokens, max(inst.cache_ids) + 1)):
                    cache_is_enough = False
            elif isinstance(inst, PullCache):
                if inst.virtual_image_cache and rcb.virtual_image_cache is None and self.context.image_cache_block_manager:
                    rcb.virtual_image_cache = self.context.image_cache_block_manager.allocate_virtual_cache()
                    if not self.context.image_cache_block_manager.realloc(rcb.virtual_image_cache, inst.virtual_image_cache.n_cache_tokens):
                        cache_is_enough = False
                if inst.virtual_kv_cache and rcb.virtual_kv_cache is None and self.context.kv_cache_block_manager:
                    rcb.virtual_kv_cache = self.context.kv_cache_block_manager.allocate_virtual_cache()
                    if not self.context.kv_cache_block_manager.realloc(rcb.virtual_kv_cache, inst.virtual_kv_cache.n_cache_tokens):
                        cache_is_enough = False
            if cache_is_enough:
                after_allocated_cache.append(rcb)
            else:
                self.free_request(rcb)
            
                
        # 2. classify seqs
        for rcb in after_allocated_cache:
            inst = rcb.current_instruction()
            if isinstance(inst, Fill):
                if len(inst.token_ids) == 1:
                    decode_seqs.append(rcb)
                else:
                    prefill_seqs.append(rcb)
            elif isinstance(inst, ImageEmbed):
                embed_seqs.append(rcb)
            else:
                this_step.append(rcb)

        # 2. batch image embed
        for rcb in embed_seqs:
            if batch_embed_images < self.image_budgets:
                this_step.append(rcb)
                batch_embed_images += 1 # todo cope with multi image
            else:
                next_step.append(rcb)

        # 3. batch prefill and decode
        fill_seqs = prefill_seqs + decode_seqs if self.config.priority == 'prefill' else decode_seqs + prefill_seqs
            
        for rcb in fill_seqs:
            inst = rcb.current_instruction()
            n_tokens = len(inst.token_ids)
            if batch_fill_tokens + n_tokens <= self.token_budgets:
                this_step.append(rcb)
                batch_fill_tokens += n_tokens
            elif batch_fill_tokens < self.token_budgets and n_tokens > 1 and self.config.chunked_prefill and (isinstance(inst, TextFill) or isinstance(inst, ImageEmbedFill)): # if it is prefill and we can chunk part of it
                # todo chunk_size align to block_size to make prefix cache
                chunk_size = self.token_budgets - batch_fill_tokens
                inst.chunk_prefill(chunk_size)
                this_step.append(rcb)
                batch_fill_tokens += chunk_size
            elif batch_fill_tokens == 0: # avoid the prefill larger than token_budgets block
                this_step.append(rcb)
                batch_fill_tokens += n_tokens
            else:
                next_step.append(rcb)

        if self.config.debug:
            logger.debug(f'------------------------------ scheduler step {self.step_cnt} ------------------------------')
            logger.debug(f'inst: ' + ' '.join(f'{rcb.current_instruction()}' for rcb in this_step))
            logger.debug(f'batch images {batch_embed_images}')
            logger.debug(f'batch tokens {batch_fill_tokens}')

        self.running = next_step
        return BatchRequest(this_step)

    def get_metrics(self) -> BatchSchedulerMetrics:
        return BatchSchedulerMetrics(
            n_running=self.running_cnt, 
            n_new=self.waiting.qsize(),
            n_migrating=len(self.migrating_requests), 
            n_waiting_to_pull=self.pulling.qsize(), 
        )
