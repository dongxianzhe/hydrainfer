import torch
from torch import Tensor
from typing import Optional

class InputParameters:
    def __init__(self,
            cache_length,
            num_sequences: int = 0,
            empty_kv_cache_for_all_sequences: bool=True,
            q_cu_seq_lens: Optional[Tensor]=None,
            kv_cu_seq_lens: Optional[Tensor]=None, 
            new_cache_slots: Optional[Tensor]=None,
            block_tables: Optional[Tensor]=None,
            cu_blocks_lens: Optional[Tensor]=None
           ):
        self.cache_length = cache_length
        # todo remove fixed size kv cache

        # total number of sequences in the batch
        self.num_sequences = 0
        # whether the kv cache is empty for all sequences
        self.empty_kv_cache_for_all_sequences = True
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
    
    def to(device: torch.device):
        if q_cu_seq_lens   is not None: 
            self.q_cu_seq_lens   = q_cu_seq_lens.to(device)
        if kv_cu_seq_lens  is not None: 
            self.kv_cu_seq_lens  = kv_cu_seq_lens.to(device)
        if new_cache_slots is not None: 
            self.new_cache_slots = new_cache_slots.to(device)
        if block_tables    is not None: 
            self.block_tables    = block_tables.to(device)
        if cu_blocks_lens  is not None: 
            self.cu_blocks_lens  = cu_blocks_lens.to(device)