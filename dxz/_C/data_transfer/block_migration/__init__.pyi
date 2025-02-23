from torch import Tensor
from typing import List

cudaMemoryIpcHandle = List[int]

def get_ipc_mem_handle(tensor: Tensor) -> cudaMemoryIpcHandle:
    """
    return type list[int] is cudaIpcMemHandle_t
    """
    ...

def migrate_blocks(
	src_block_table: list[int], 
	dst_block_table: list[int], 
	src_cache: cudaMemoryIpcHandle, 
	dst_cache: Tensor, # (n_layers, n_tokens, n_blocks, block_size, n_heads, head_size)
):
	...