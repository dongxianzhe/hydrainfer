import math
import torch
from torch import nn, Tensor
from typing import Optional, Union
from dxz.memory.kv_cache import KVCache
from dxz.utils.statistic import attention_score_heatmap, histogram
from dxz.utils import attention_utils
from dxz.utils.attention_utils import sparsity
from dataclasses import dataclass
from dxz.utils.logger import getLogger
logger = getLogger(__name__)

try:
    from dxz._C.kernel.flash_attn import mha_varlen_fwd
except ImportError:
    logger.warning('compiled flash attention mha_varlen_fwd import failed')
    mha_varlen_fwd = None


"""
we use chain of responsibility to design attention module considering the following requirements:
1. some kernel may be unavailable because of compilation failure, library installation failure.
2. some optimization need attention score to evict kvcache or prune token but fused kernel can't get intermediate attention score.
3. some kernel can't work on all dtype, hardware version.
4. we may want to dynamically select best kernel according to batch information and schedule policy
5. we may want to statically configure a specific attention kernel
so we design attention handlers for each kernel, and link them to a linked list data structure, each handler will check weather it can cope with the situation if it can, it will compute and return the result, if it can't, it will call the next handler to cope with it. at last the request must be handled becuase we have an pytorch implemented handler, however it is very slow.
"""


@dataclass
class AttentionParameters:
    # each layer have an AttentionParameters

    # Typically each layer has its own kvcache, 
    # but consider that in token pruning there may be different for each layer and all layer shared one kvcache or some layers shared one kvcache
    kv_cache: KVCache
    # cumulative new token length of each sequence
    # IntTensor: (n_tokens, )
    # eg. if we have a sequence in prefill stage with 512 tokens and a sequence in decode stage with one token, then the q_cu_seq_lens will be [0, 512, 513]
    # IntTensor: (n_tokens + 1, )
    q_cu_seq_lens: Tensor = None
    # tokens in cache + new tokens
    # eg. if we have a sequence in prefill stage with 512 tokens and a sequence with prompt 512 tokens and allready has 1024 tokens, then the q_cu_seq_lens will be [0, 512, 512 + 1]
    # k_cu_seq_lens will be [0, 512, 512 + 1024 + 1]
    # IntTensor: (n_tokens + 1, )
    kv_cu_seq_lens: Tensor = None
    # the last block length of kv cache for each sequence
    paged_kv_last_page_len: Tensor = None
    # logical cache slot for each new token
    # used to store kv cache to right slot/block
    # IntTensor: (n_tokens, )
    new_cache_slots: Tensor = None
    # block ids for each sequence flattened into 1D tensor
    # IntTensor (n_blocks, )
    block_tables: Tensor = None
    # cumulative block length for each sequcne
    # IntTensor (n_seq + 1, )
    cu_blocks_lens: Tensor  = None
    # total number of sequences in the batch
    num_sequences: int = None
    all_sequences_decode: bool = False
    # maximum sequence length for query and kv
    # used to help dispatch chooling the right kernel based on the length
    q_max_seq_len: int = 128
    kv_max_seq_len: int = 128
    # because flashinfer interface must plan in advance to get full performance, so we let the engine control whether to use flashinfer
    flash_infer_handler: Optional[Union["flashinfer.BatchPrefillWithPagedKVCacheWrapper", "flashinfer.BatchDecodeWithPagedKVCacheWrapper"]] = None
    def to(self, device: torch.device):
        if self.q_cu_seq_lens is not None:
            self.q_cu_seq_lens = self.q_cu_seq_lens.to(device)
        if self.kv_cu_seq_lens is not None:
            self.kv_cu_seq_lens = self.kv_cu_seq_lens.to(device)
        if self.paged_kv_last_page_len is not None:
            self.paged_kv_last_page_len = self.paged_kv_last_page_len.to(device)
        if self.new_cache_slots is not None:
            self.new_cache_slots = self.new_cache_slots.to(device)
        if self.block_tables is not None:
            self.block_tables = self.block_tables.to(device)
        if self.cu_blocks_lens is not None:
            self.cu_blocks_lens = self.cu_blocks_lens.to(device)
    def print(self):
        # print params info used in debug
        print('============================= attention params ===============================')
        if self.q_cu_seq_lens is not None:
            print(f'self.q_cu_seq_lens         [0:8]: {self.q_cu_seq_lens[0:8]}')
        if self.kv_cu_seq_lens is not None:
            print(f'self.kv_cu_seq_lens        [0:8]: {self.kv_cu_seq_lens[0:8]}')
        if self.paged_kv_last_page_len is not None:
            print(f'self.paged_kv_last_page_len[0:8]: {self.paged_kv_last_page_len[0:8]}')
        if self.new_cache_slots is not None:
            print(f'self.new_cache_slots       [0:8]: {self.new_cache_slots[0:8]}')
        if self.block_tables is not None:
            print(f'self.block_tables          [0:8]: {self.block_tables[0:8]}')
        if self.cu_blocks_lens is not None:
            print(f'self.cu_blocks_lens        [0:8]: {self.cu_blocks_lens[0:8]}')
        if self.num_sequences is not None:
            print(f'self.num_sequences              : {self.num_sequences}')
        if self.all_sequences_decode is not None:
            print(f'self.all_sequences_decode       : {self.all_sequences_decode}')
        if self.q_max_seq_len is not None:
            print(f'self.q_max_seq_len              : {self.q_max_seq_len}')
        if self.kv_max_seq_len is not None:
            print(f'self.kv_max_seq_len             : {self.kv_max_seq_len}')
        if self.flash_infer_handler is not None:
            print(f'self.flash_infer_handler        : {self.flash_infer_handler}')
        print('==============================================================================')


class AttentionParametersBuilder:
    """
    usage
    1. init builder with block_size and device and flash infer handler
    2. add new request with necessary attention control data
    3. add kv cache
    4. build params. all kv cache will have same attention control tensor
    """
    def __init__(self,
                num_qo_heads: int,
                num_kv_heads: int,
                head_dim: int, 
                block_size: int,
                device: torch.device, 
                flash_infer_batch_prefill_handler: Optional["flashinfer.BatchPrefillWithPagedKVCacheWrapper"] = None, 
                flash_infer_batch_decode_handler: Optional["flashinfer.BatchDecodeWithPagedKVCacheWrapper"] = None
                ):
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.device = device
        self.flash_infer_batch_prefill_handler = flash_infer_batch_prefill_handler
        self.flash_infer_batch_decode_handler  = flash_infer_batch_decode_handler

        self.kv_caches: list[KVCache] = []
        self.q_cu_seq_lens: list[int] = [0]
        self.kv_cu_seq_lens: list[int] = [0]
        self.paged_kv_last_page_len: list[int] = []
        self.new_cache_slots: list[int] = []
        self.block_tables: list[int] = []
        self.cu_blocks_lens: list[int] = [0]
        self.num_sequences: int = 0
        self.all_sequences_decode: bool = True
        self.q_max_seq_len: int = 0
        self.kv_max_seq_len: int = 0

    def add_request(self, q_seq_len: int, kv_seq_len:  int, new_cache_slots: list[int], block_table: list[int]):
        self.q_cu_seq_lens.append(self.q_cu_seq_lens[-1] + q_seq_len)
        self.kv_cu_seq_lens.append(self.kv_cu_seq_lens[-1] + kv_seq_len)
        self.paged_kv_last_page_len.append((kv_seq_len + self.block_size - 1) % self.block_size + 1)
        self.new_cache_slots += new_cache_slots
        self.block_tables += block_table
        self.cu_blocks_lens.append(self.cu_blocks_lens[-1] + len(block_table))
        self.num_sequences += 1
        self.all_sequences_decode = self.all_sequences_decode and q_seq_len == 1
        self.q_max_seq_len = max(self.q_max_seq_len, q_seq_len)
        self.kv_max_seq_len = max(self.kv_max_seq_len, kv_seq_len)

    def add_kv_cache(self, kv_cache: KVCache):
        self.kv_caches.append(kv_cache)
    
    def build_attention_parameters(self) -> list[AttentionParameters]:
        q_cu_seq_lens = torch.tensor(self.q_cu_seq_lens, dtype=torch.int, device=self.device)
        kv_cu_seq_lens = torch.tensor(self.kv_cu_seq_lens, dtype=torch.int, device=self.device)
        paged_kv_last_page_len = torch.tensor(self.paged_kv_last_page_len, dtype=torch.int, device=self.device)
        new_cache_slots = torch.tensor(self.new_cache_slots, dtype=torch.int, device=self.device)
        block_tables = torch.tensor(self.block_tables, dtype=torch.int, device=self.device)
        cu_blocks_lens = torch.tensor(self.cu_blocks_lens, dtype=torch.int, device=self.device) 

        flash_infer_handler = None
        if self.flash_infer_batch_decode_handler is not None and self.all_sequences_decode:
            self.flash_infer_batch_decode_handler.plan(
                indptr = cu_blocks_lens, 
                indices = block_tables, 
                last_page_len = paged_kv_last_page_len,
                num_qo_heads = self.num_qo_heads,
                num_kv_heads = self.num_kv_heads,
                head_dim = self.head_dim, 
                page_size = self.block_size,
            )
            flash_infer_handler = self.flash_infer_batch_decode_handler
        elif self.flash_infer_batch_prefill_handler is not None and not self.all_sequences_decode:
            self.flash_infer_batch_prefill_handler.plan(
                qo_indptr = q_cu_seq_lens, 
                paged_kv_indptr = cu_blocks_lens, 
                paged_kv_indices = block_tables, 
                paged_kv_last_page_len = paged_kv_last_page_len,
                num_qo_heads = self.num_qo_heads,
                num_kv_heads = self.num_kv_heads,
                head_dim = self.head_dim, 
                page_size = self.block_size,
                causal=True
            )
            flash_infer_handler = self.flash_infer_batch_prefill_handler

        return [AttentionParameters(
            kv_cache = kv_cache, 
            q_cu_seq_lens = q_cu_seq_lens, 
            kv_cu_seq_lens = kv_cu_seq_lens, 
            paged_kv_last_page_len = paged_kv_last_page_len, 
            new_cache_slots = new_cache_slots, 
            block_tables = block_tables, 
            cu_blocks_lens = cu_blocks_lens, 
            num_sequences = self.num_sequences, 
            all_sequences_decode = self.all_sequences_decode, 
            q_max_seq_len = self.q_max_seq_len, 
            kv_max_seq_len = self.kv_max_seq_len, 
            flash_infer_handler = flash_infer_handler
        ) for kv_cache in self.kv_caches]


@dataclass
class CausalGroupedQueryPageAttentionConfig:
    n_qo_heads: int
    n_kv_heads: int
    head_dim: int


@dataclass
class CausalGroupedQueryPageAttentionOutput:
    o: Tensor


class FlashInferCausalGroupedQueryPageAttentionHandler(nn.Module):
    def __init__(self, config: CausalGroupedQueryPageAttentionConfig):
        super().__init__()
        assert config.n_qo_heads % config.n_kv_heads == 0, f"n_qo_heads {config.n_qo_heads} is not divisible by n_kv_heads {config.n_kv_heads}"
        self.n_qo_heads     = config.n_qo_heads
        self.n_kv_heads     = config.n_kv_heads
        self.head_dim       = config.head_dim

        self.next_handler: nn.Module = None

    def forward(self, query: Tensor, attention_params: AttentionParameters) -> CausalGroupedQueryPageAttentionOutput:
        # query (n_tokens, n_qo_heads * head_dim)
        # o (n_tokens, n_heads, head_dim)
        if query.device == torch.device('cpu'):
            return self.next_handler(query, attention_params)

        if attention_params.flash_infer_handler is None:
            return self.next_handler(query, attention_params)

        kv_cache = attention_params.kv_cache
        key_cache, value_cache = kv_cache.get_kv_cache()
        output = attention_params.flash_infer_handler.run(query, (key_cache, value_cache))
        output = output.view(-1, self.n_qo_heads * self.head_dim)

        return CausalGroupedQueryPageAttentionOutput(o = output)


class FlashAttentionCausalGroupedQueryPageAttentionHandler(nn.Module):
    def __init__(self, config: CausalGroupedQueryPageAttentionConfig):
        super().__init__()
        assert config.n_qo_heads % config.n_kv_heads == 0, f"n_qo_heads {config.n_qo_heads} is not divisible by n_kv_heads {config.n_kv_heads}"
        self.n_qo_heads     = config.n_qo_heads
        self.n_kv_heads     = config.n_kv_heads
        self.head_dim       = config.head_dim

        self.next_handler: nn.Module = None

    def forward(self, query: Tensor, attention_params: AttentionParameters) -> CausalGroupedQueryPageAttentionOutput:
        # query (n_tokens, n_qo_heads * head_dim)
        # o (n_tokens, n_heads, head_dim)
        if query.device == torch.device('cpu'):
            return self.next_handler(query, attention_params)
        if mha_varlen_fwd is None:
            return self.next_handler(query, attention_params)

        kv_cache = attention_params.kv_cache
        key_cache, value_cache = kv_cache.get_kv_cache()

        output=torch.empty_like(query)
        mha_varlen_fwd(
            output,
            query, 
            key_cache, 
            value_cache, 
            attention_params.q_cu_seq_lens, 
            attention_params.kv_cu_seq_lens, 
            attention_params.block_tables, 
            attention_params.cu_blocks_lens, 
            None,
            attention_params.q_max_seq_len,
            attention_params.kv_max_seq_len,
            1. / math.sqrt(self.head_dim),
            0,
            -1,
            0,
            0
        )
        output = output.view(-1, self.n_qo_heads * self.head_dim)

        return CausalGroupedQueryPageAttentionOutput(o = output)


class TorchCausalGroupedQueryPageAttentionHandler(nn.Module):
    def __init__(self, config: CausalGroupedQueryPageAttentionConfig):
        super().__init__()
        assert config.n_qo_heads % config.n_kv_heads == 0, f"n_qo_heads {config.n_qo_heads} is not divisible by n_kv_heads {config.n_kv_heads}"
        self.n_qo_heads     = config.n_qo_heads
        self.n_kv_heads     = config.n_kv_heads
        self.head_dim       = config.head_dim

        self.next_handler: nn.Module = None

    def forward(self, query: Tensor, attention_params: AttentionParameters) -> CausalGroupedQueryPageAttentionOutput:
        # query (n_tokens, n_qo_heads * head_dim)
        # o (n_tokens, n_heads, head_dim)
        kv_cache = attention_params.kv_cache
        key_cache, value_cache = kv_cache.get_kv_cache()

        outputs = []
        for i in range(attention_params.num_sequences):
            block_table = attention_params.block_tables[attention_params.cu_blocks_lens[i]: attention_params.cu_blocks_lens[i + 1]]
            key = key_cache[block_table, :, :, :].reshape(-1, self.n_kv_heads, self.head_dim)
            value = value_cache[block_table, :, :, :].reshape(-1, self.n_kv_heads, self.head_dim)
            k = key[: attention_params.kv_cu_seq_lens[i + 1] - attention_params.kv_cu_seq_lens[i], :, :].to(torch.float)
            v = value[: attention_params.kv_cu_seq_lens[i + 1] - attention_params.kv_cu_seq_lens[i],:, :].to(torch.float)
            q = query[attention_params.q_cu_seq_lens[i]: attention_params.q_cu_seq_lens[i + 1], :, :].to(torch.float)
            # q (qo_seq_len, n_qo_heads, head_dim)
            # k (kv_seq_len, n_kv_heads, head_dim)
            # v (kv_seq_len, n_kv_heads, head_dim)
            group_size = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(repeats=group_size, dim=1)
            v = v.repeat_interleave(repeats=group_size, dim=1)
            # k (kv_seq_len, n_qo_heads, head_dim)
            # v (kv_seq_len, n_qo_heads, head_dim)

            # compute score
            scores = torch.einsum('qhd,khd->hqk', q, k)

            # sm_scale
            sm_scale = 1. / math.sqrt(self.head_dim)
            scores *= sm_scale

            # mask
            num_heads, q_seq_len, k_seq_len = scores.shape
            x = torch.arange(k_seq_len, device=q.device)[None, None, :].repeat(num_heads, q_seq_len, 1)
            y = torch.arange(q_seq_len, device=q.device)[None, :, None].repeat(num_heads, 1, k_seq_len)
            mask = (x - y) > (k_seq_len - q_seq_len)
            scores.masked_fill_(mask=mask, value=float('-inf'))

            # # save attention score data
            # if scores.shape[1] == scores.shape[2]:
            #     torch.save(scores, f'layer{attention_params.layer_id}.pt')

            # # plot atention score
            # if scores.shape[1] == scores.shape[2]: # prefill stage
            #     print(f'layer{attention_params.layer_id} scores.shape {scores.shape}')

            #     attention_utils.sparsity(scores, mask, 0)
            #     for head_id in range(scores.shape[0]):
            #         attention_score_heatmap(score=scores[head_id], name=f'attention_score_layer{attention_params.layer_id}_head{head_id}', fig_size=50)

            # # meausure sparsity
            # if scores.shape[1] == scores.shape[2]:
            #     print(f'sparsity {sparsity(scores, mask, 0.01)}')

            # # plot score distrubtion
            # if scores.shape[1] == scores.shape[2]:
            #     print(f'scores.shape {scores.shape}')
            #     histogram(data=scores, fig_size=5, range=(-25, 10), bins=100, name=f'distribution_attention_score_layer{attention_params.layer_id}')

            # softmax
            scores = torch.softmax(scores, dim=-1)
            o = torch.einsum("hqk,khd->qhd", scores, v)
            outputs.append(o)

        output = torch.cat(outputs, dim=0)
        output = output.view(-1, self.n_qo_heads * self.head_dim)
        output = output.to(query.dtype)

        return CausalGroupedQueryPageAttentionOutput(o = output)


class CausalGroupedQueryPageAttention(nn.Module):
    def __init__(self, config: CausalGroupedQueryPageAttentionConfig):
        super().__init__()
        assert config.n_qo_heads % config.n_kv_heads == 0, f"n_qo_heads {config.n_qo_heads} is not divisible by n_kv_heads {config.n_kv_heads}"
        self.n_qo_heads     = config.n_qo_heads
        self.n_kv_heads     = config.n_kv_heads
        self.head_dim       = config.head_dim

        self.handlers = [
            FlashInferCausalGroupedQueryPageAttentionHandler(config), 
            FlashAttentionCausalGroupedQueryPageAttentionHandler(config), 
            TorchCausalGroupedQueryPageAttentionHandler(config), 
        ]
        for i in range(len(self.handlers) - 1):
            self.handlers[i].next_handler = self.handlers[i + 1]
        self.handler = self.handlers[0]

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attention_params: AttentionParameters) -> CausalGroupedQueryPageAttentionOutput:
        # 1. reshape queyy key value shape
        n_tokens = query.shape[0]
        query = query.view(n_tokens, self.n_qo_heads, self.head_dim)
        key = key.view(n_tokens, self.n_kv_heads, self.head_dim)
        value = value.view(n_tokens, self.n_kv_heads, self.head_dim)

        # 2. append new kv cache
        kv_cache = attention_params.kv_cache
        kv_cache.set_kv_cache(attention_params.new_cache_slots, key,value)

        # 3. compute attention
        return self.handler(query, attention_params)