from torch import Tensor
from typing import List

cudaMemoryIpcHandle = List[int]

def get_ipc_mem_handle(tensor: Tensor) -> cudaMemoryIpcHandle:
    """
    return type list[int] is cudaIpcMemHandle_t
    """
    ...

def register_ipc_mem_handle(kv_cache_handle_vec: cudaMemoryIpcHandle) -> int:
    """
    return dev_ptr_index used in migrate_blocks
    """
    ...

def migrate_blocks(
	prefill_start_head: int,
	prefill_end_head: int,
	
	prefill_block_indexes: list[int],

	decoding_start_head: int,
	decoding_end_head:int,

	decoding_block_indexes: list[int],

	prefill_dev_ptr_index: int,
	num_heads: int,

	decoding_worker_kv_cache: Tensor
	# [2, num_blocks, block_size, num_kv_heads, head_size]
):
    ...