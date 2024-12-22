import torch
from torch import Tensor
from typing import Optional
from dxz.memory.kv_cache import KVCache


class InputParameters:
    """
    if layers have different kv cache management, layer_input_params will not be none and other params be none

    if all layers have same kv cache management, layer_input_params will be none, and others will not be none
    """
    def __init__(self,
            num_sequences: int = 0,
            q_cu_seq_lens: Optional[Tensor]=None,
            kv_cu_seq_lens: Optional[Tensor]=None, 
            new_cache_slots: Optional[Tensor]=None,
            block_tables: Optional[Tensor]=None,
            cu_blocks_lens: Optional[Tensor]=None, 
            q_max_seq_len: int=1024, 
            kv_max_seq_len: int=1024, 
            layer_id: int = 0,
            layer_input_params: list["InputParameters"] = None
           ):

        # total number of sequences in the batch
        self.num_sequences = num_sequences
        # cumulative sequence length of each sequence
        # used to determine the token range for each sequence
        # IntTensor: (n_seq + 1)
        # for example 3 sequences with length 2, 3, 4
        # the cu_seq_lens is [0, 2, 5, 9]
        self.q_cu_seq_lens = q_cu_seq_lens # query len
        self.kv_cu_seq_lens = kv_cu_seq_lens # tokens in cache + new tokens
        # logical cache slot for each new token
        # used to store kv cache to right slot/block
        # IntTensor: (n_tokens, )
        self.new_cache_slots = new_cache_slots
        # block ids for each sequence flattened into 1D tensor
        # IntTensor (n_blocks, )
        self.block_tables = block_tables 
        # cumulative block length for each sequcne
        # IntTensor (n_seq + 1, )
        self.cu_blocks_lens = cu_blocks_lens
        # maximum sequence length for query and kv
        # used to help dispatch chooling the right kernel based on the length
        self.q_max_seq_len = q_max_seq_len
        self.kv_max_seq_len = kv_max_seq_len

        self.layer_id = layer_id # set when foward or in layer_input_params
        self.layer_input_params = layer_input_params
    
    def to(self, device: torch.device):
        if self.q_cu_seq_lens   is not None: 
            self.q_cu_seq_lens   = self.q_cu_seq_lens.to(device)
        if self.kv_cu_seq_lens  is not None: 
            self.kv_cu_seq_lens  = self.kv_cu_seq_lens.to(device)
        if self.new_cache_slots is not None: 
            self.new_cache_slots = self.new_cache_slots.to(device)
        if self.block_tables    is not None: 
            self.block_tables    = self.block_tables.to(device)
        if self.cu_blocks_lens  is not None: 
            self.cu_blocks_lens  = self.cu_blocks_lens.to(device)
        if self.layer_input_params is not None:
            for input_params in self.layer_input_params:
                input_params.to(device)

    def print(self):
        print(f'-----------------------------input_params-------------------------------')
        if self.layer_input_params is None:
            print(f'num_sequences        : {self.num_sequences}')
            print(f'q_cu_seq_lens.shape  : {self.q_cu_seq_lens.shape}')
            print(f'kv_cu_seq_lens.shape : {self.kv_cu_seq_lens.shape}')
            print(f'new_cache_slots.shape: {self.new_cache_slots.shape}')
            print(f'block_tables.shape   : {self.block_tables.shape}')
            print(f'cu_blocks_lens.shape : {self.cu_blocks_lens.shape}')
            print(f'q_cu_seq_lens[:8]    : {self.q_cu_seq_lens[:8]}')
            print(f'kv_cu_seq_lens[:8]   : {self.kv_cu_seq_lens[:8]}')
            print(f'new_cache_slots[:8]  : {self.new_cache_slots[:8]}')
            print(f'block_tables[:8]     : {self.block_tables[:8]}')
            print(f'cu_blocks_lens[:8]   : {self.cu_blocks_lens[:8]}')
        else:
            for layer_id, input_params in enumerate(self.layer_input_params):
                print(f'------------------------layer{layer_id}--------------------------')
                print(f'num_sequences        : {input_params.num_sequences}')
                print(f'q_cu_seq_lens.shape  : {input_params.q_cu_seq_lens.shape}')
                print(f'kv_cu_seq_lens.shape : {input_params.kv_cu_seq_lens.shape}')
                print(f'new_cache_slots.shape: {input_params.new_cache_slots.shape}')
                print(f'block_tables.shape   : {input_params.block_tables.shape}')
                print(f'cu_blocks_lens.shape : {input_params.cu_blocks_lens.shape}')
                print(f'q_cu_seq_lens[:8]    : {input_params.q_cu_seq_lens[:8]}')
                print(f'kv_cu_seq_lens[:8]   : {input_params.kv_cu_seq_lens[:8]}')
                print(f'new_cache_slots[:8]  : {input_params.new_cache_slots[:8]}')
                print(f'block_tables[:8]     : {input_params.block_tables[:8]}')
                print(f'cu_blocks_lens[:8]   : {input_params.cu_blocks_lens[:8]}')
        print(f'-----------------------------input_params-------------------------------')

from dataclasses import dataclass
@dataclass
class AttentionParameters:
    kv_cache: KVCache
    q_cu_seq_lens: Tensor = None
    kv_cu_seq_lens: Tensor = None
    paged_kv_last_page_len: Tensor = None
    new_cache_slots: Tensor = None
    block_tables: Tensor = None
    cu_blocks_lens: Tensor  = None
    num_sequences: int = None
    all_sequences_decode: bool = False
    q_max_seq_len: int = 128
    kv_max_seq_len: int = 128
    flash_infer_handler: "flashinfer.BatchPrefillWithPagedKVCacheWrapper" = None

@dataclass
class VisionModelParameters:
    return_last_layer_attention: bool = False

class VisionModelOutput:
    scores: Tensor = None

@dataclass
class ModelParameters:
    attention_params: Optional[list[AttentionParameters]] = None
    all_sequences_decode: bool = False
    selected_token_ids: list[int] = None
    embed_token_pruning_params: Optional[dict] = None
    prefill_token_pruning_params: Optional[dict] = None
    decode_kvcache_eviction_parmas: Optional[dict] = None
    vision_params: VisionModelParameters = None
