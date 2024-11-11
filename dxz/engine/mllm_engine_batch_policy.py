from torch import Tensor
import queue
from dataclasses import dataclass, field
from PIL import Image
from dxz.memory.block_allocator import BlockAllocator

@dataclass
class MultiModalSequence:
    prompt: str = ""
    token_ids: list[int] = field(default_factory=list)
    num_prompt_tokens: int = 0 # the number of prompt tokens
    n_kv_cache_tokens: int = 0 # the number of tokens already in kv cache
    block_table: list[int] = field(default_factory=list)
    pixel_values: Tensor = None
    # stop criterian
    max_tokens: int = 50
    eos_token_id:int = 0
    max_seq_len:int = 1024

    @property
    def is_finished(self) -> bool:
        return self.token_ids[-1] == self.eos_token_id or (len(self.token_ids) - self.num_prompt_tokens) == self.max_tokens or len(self.token_ids) > self.max_seq_len

    @property
    def is_prefill(self) -> bool:
        return self.n_kv_cache_tokens < self.num_prompt_tokens
    
    @property
    def is_prefill_but_not_new(self) -> bool:
        return self.n_kv_cache_tokens < self.num_prompt_tokens and not self.is_new

    @property
    def is_decode(self) -> bool:
        return self.n_kv_cache_tokens >= self.num_prompt_tokens
    
    @property
    def is_decode_but_not_finished(self) -> bool:
        return self.n_kv_cache_tokens >= self.num_prompt_tokens and not self.is_finished

    @property
    def is_new(self) -> bool:
        return self.n_kv_cache_tokens == 0

class NoBatchPolicy:
    def __init__(self, allocator: BlockAllocator, block_size: int):
        # this is a no-batch policy, which means we only have one sequence at a time
        self.new_queue = queue.Queue()
        self.intermediate_sequence = None # the sequence which is prefilling or decoding
        self.allocator = allocator
        self.block_size = block_size
    
    def add_new_sequence(self, sequence: MultiModalSequence):
        self.new_queue.put(sequence)

    def add_intermediate_sequences(self, batch_sequences: list[MultiModalSequence]):
        assert len(batch_sequences) == 1, "no batch policy only support batch size = 1"
        sequence = batch_sequences[0]
        if sequence.is_finished:
            self.intermediate_sequence = None
            self.allocator.free(sequence.block_table)
            sequence.block_table = []
        else:
            self.intermediate_sequence = sequence

    def batch(self) -> tuple[list[MultiModalSequence], list[int], list[Tensor]]:
        batch_sequences: list[MultiModalSequence] = []
        q_seq_lens: list[int] = []
        pixel_values: list[Tensor] = []
        # 1. select a sequence
        sequence = None
        if self.intermediate_sequence is None:
            if not self.new_queue.empty():
                sequence = self.new_queue.get()
        else:
            sequence = self.intermediate_sequence
        if sequence is None:
            return batch_sequences, q_seq_lens, pixel_values

        # 2. allocate block
        if len(sequence.block_table) * self.block_size < len(sequence.token_ids):
            n_blocks = (len(sequence.token_ids) - len(sequence.block_table) * self.block_size + self.block_size - 1) // self.block_size
            blocks = self.allocator.allocate(n_blocks)
            assert len(blocks) > 0, "No more blocks available"
            sequence.block_table += blocks

        batch_sequences.append(sequence)
        q_seq_lens.append(len(sequence.token_ids) - sequence.n_kv_cache_tokens)
        if sequence.is_new:
            pixel_values.append(sequence.pixel_values)
        return batch_sequences, q_seq_lens, pixel_values