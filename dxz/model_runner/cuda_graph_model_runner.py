import torch
from torch import nn, Tensor
import torch.types
from dxz.memory.kv_cache import KVCache
from dxz.model.parameters import InputParameters

class CudaGraphModelRunner:
    def __init__(self, model_runner, dtype:torch.types, device: torch.device, block_size:int, vocab_size: int, kv_caches: list[KVCache], cuda_graph_max_batch_size:int=64, cuda_graph_max_seq_len:int=1024):
        self.model_runner = model_runner
        self.dtype = dtype
        self.device = device
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.kv_caches = kv_caches

        cuda_graph_max_block_len_per_seq = (cuda_graph_max_seq_len + self.block_size - 1) // self.block_size
        cuda_graph_max_block_len = cuda_graph_max_batch_size * cuda_graph_max_block_len_per_seq

        self.graphs: dict[int, torch.cuda.CUDAGraph] = {} # batch_size -> cuda graph
        self.static_input_ids       = torch.empty(cuda_graph_max_batch_size    , dtype=torch.int, device=self.device)
        self.static_position_ids    = torch.empty(cuda_graph_max_batch_size    , dtype=torch.int, device=self.device)
        self.static_q_cu_seq_lens   = torch.empty(cuda_graph_max_batch_size + 1, dtype=torch.int, device=self.device)
        self.static_kv_cu_seq_lens  = torch.empty(cuda_graph_max_batch_size + 1, dtype=torch.int, device=self.device)
        self.static_new_cache_slots = torch.empty(cuda_graph_max_batch_size    , dtype=torch.int, device=self.device)
        self.static_block_tables    = torch.empty(cuda_graph_max_block_len     , dtype=torch.int, device=self.device)
        self.static_cu_blocks_lens  = torch.empty(cuda_graph_max_batch_size + 1, dtype=torch.int, device=self.device)
        self.static_logits          = torch.empty((cuda_graph_max_batch_size   , self.vocab_size), dtype=self.dtype, device=self.device)

        batch_sizes = list(range(1, cuda_graph_max_batch_size, 1))
        print(f'cuda graph capture batch_sizes {batch_sizes}')
        for batch_size in batch_sizes:
            input_ids    = self.static_input_ids[:batch_size]
            position_ids = self.static_position_ids[:batch_size]
            input_params = InputParameters(
                num_sequences   = batch_size, 
                q_cu_seq_lens   = self.static_q_cu_seq_lens[:batch_size+1], 
                kv_cu_seq_lens  = self.static_kv_cu_seq_lens[:batch_size+1], 
                new_cache_slots = self.static_new_cache_slots[:batch_size], 
                block_tables    = self.static_block_tables, 
                cu_blocks_lens  = self.static_cu_blocks_lens[:batch_size+1]
            )
            # Run the model a few times without capturing the graph.
            # This is to make sure that the captured graph does not include the
            # kernel launches for initial benchmarking (e.g., Triton autotune).
            # Note one iteration is not enough for torch.jit.script
            self.model_runner(input_ids, position_ids, self.kv_caches, input_params)
            self.model_runner(input_ids, position_ids, self.kv_caches, input_params)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self.static_logits[:batch_size] = self.model_runner(input_ids, position_ids, self.kv_caches, input_params)
            torch.cuda.synchronize()
            self.graphs[batch_size] = g

    def forward(self, input_ids: Tensor, position_ids: Tensor, kv_caches: list[KVCache], input_params: InputParameters) -> Tensor:
        all_sequence_is_decode: bool = input_params.num_sequences == input_ids.numel()
        if all_sequence_is_decode and input_params.num_sequences in self.graphs:
            g = self.graphs[input_params.num_sequences]
            self.static_input_ids      [: input_params.num_sequences         ].copy_(input_ids)
            self.static_position_ids   [: input_params.num_sequences         ].copy_(position_ids)
            self.static_q_cu_seq_lens  [: input_params.num_sequences + 1     ].copy_(input_params.q_cu_seq_lens)
            self.static_kv_cu_seq_lens [: input_params.num_sequences + 1     ].copy_(input_params.kv_cu_seq_lens)
            self.static_new_cache_slots[: input_params.num_sequences         ].copy_(input_params.new_cache_slots)
            self.static_block_tables   [: input_params.block_tables.shape[-1]].copy_(input_params.block_tables)
            self.static_cu_blocks_lens [: input_params.num_sequences + 1     ].copy_(input_params.cu_blocks_lens)
            g.replay()
            logits = self.static_logits[: input_params.num_sequences, :]
            return logits
        else:
            logits = self.model_runner.forward(input_ids, position_ids, self.kv_caches, input_params)
            return logits