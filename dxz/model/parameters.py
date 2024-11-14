import torch
from torch import Tensor
from typing import Optional

class InputParameters:
    def __init__(self,
            num_sequences: int = 0,
            q_cu_seq_lens: Optional[Tensor]=None,
            kv_cu_seq_lens: Optional[Tensor]=None, 
            new_cache_slots: Optional[Tensor]=None,
            block_tables: Optional[Tensor]=None,
            cu_blocks_lens: Optional[Tensor]=None, 
            q_max_seq_len: int=1024, 
            kv_max_seq_len: int=1024
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

        self.layer_id = 0 # set when foward
    
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