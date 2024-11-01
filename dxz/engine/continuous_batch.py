import queue
from dxz.request.sequence import Sequence
from dxz.memory.block_allocator import BlockAllocator

class ContinuousBatch:
    def __init__(self, max_tokens_per_batch: int, chunk_size: int, allocator: BlockAllocator, block_size: int):
        self.max_tokens_per_batch = max_tokens_per_batch
        self.chunk_size           = chunk_size
        self.new_queue    : queue.Queue = queue.Queue()
        self.prefill_queue: queue.Queue = queue.Queue()
        self.decode_queue : queue.Queue = queue.Queue()
        self.allocator = allocator
        self.block_size = block_size
    
    def add_new_sequence(self, sequence: Sequence):
        self.new_queue.put(sequence)

    def add_prefill_sequence(self, sequence: Sequence):
        self.prefill_queue.put(sequence)
    
    def add_decode_sequence(self, sequence: Sequence):
        self.decode_queue.put(sequence)

    def batch(self):
        done: bool = False
        batch_num_tokens = 0
        batch_sequences: list[Sequence] = []
        q_seq_lens = []
        while not self.decode_queue.empty():
            sequence: Sequence = self.decode_queue.get()
            
            if len(sequence.block_table) * self.block_size < len(sequence.token_ids): # need allocate new block
                blocks = self.allocator.allocate(1)
                if len(blocks) == 0: # out of memory
                    if not self.prefill_queue.empty(): # try to find an prefill sequence to preempt
                        preempted_sequence:Sequence = self.prefill_queue.get()
                        self.allocator.free(preempted_sequence.block_table)
                        preempted_sequence.n_kv_cache_tokens = 0
                        self.new_queue.put(preempted_sequence)
                        blocks = self.allocator.allocate(1)
                    elif not self.decode_queue.empty(): # try to find an decode sequence to preempt
                        preempted_sequence:Sequence = self.prefill_queue.get()
                        self.allocator.free(preempted_sequence.block_table)
                        preempted_sequence.n_kv_cache_tokens = 0
                        self.new_queue.put(preempted_sequence)
                        blocks = self.allocator.allocate(1)
                    else: # no sequence can be preempted, stop batch
                        self.decode_queue.put(sequence)
                        done = True
                        break
                sequence.block_table += blocks
            
            if batch_num_tokens + 1 <= self.max_tokens_per_batch:
                # select this sequence
                batch_num_tokens += 1 
                batch_sequences.append(sequence)
                q_seq_lens.append(1)

            else: # this batch has too many tokens
                done = True
                break
            
        while not done and (not self.prefill_queue.empty() or not self.new_queue.empty()):
            if not self.prefill_queue.empty():
                sequence: Sequence = self.prefill_queue.get()
            else:
                sequence: Sequence = self.new_queue.get()
            # compute chunk size
            num_chunk_tokens = min(self.chunk_size, len(sequence.token_ids) - sequence.n_kv_cache_tokens, self.max_tokens_per_batch - batch_num_tokens)
            if num_chunk_tokens == 0:
                if sequence.is_new:
                    self.new_queue.put(sequence)
                else:
                    self.prefill_queue.put(sequence)
                done = True
                break
            
            if len(sequence.block_table) * self.block_size <= sequence.n_kv_cache_tokens + num_chunk_tokens: # need allocate block
                n_blocks = (sequence.n_kv_cache_tokens + num_chunk_tokens - len(sequence.block_table) * self.block_size + self.block_size - 1) // self.block_size
                blocks = self.allocator.allocate(n_blocks)
                if len(blocks) == 0: # out of memory
                    self.prefill_queue.put(sequence)
                    done = True
                    break
                else:
                    sequence.block_table += blocks

            # select this sequence
            batch_num_tokens += num_chunk_tokens
            batch_sequences.append(sequence)
            q_seq_lens.append(num_chunk_tokens)
        
        return batch_sequences, q_seq_lens