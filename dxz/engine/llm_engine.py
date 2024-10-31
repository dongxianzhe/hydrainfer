import torch
from dxz.model.parameters import InputParameters
from dxz.model.model_loader import load_model_tokenizer
from dxz.memory.kv_cache import KVCache
from dxz.request.sequence import Sequence
from dxz.memory.block_allocator import BlockAllocator
from dxz.model_runner.cuda_graph_model_runner import CudaGraphModelRunner
import queue

class LLMEngine:
    def __init__(self):
        self.device = torch.device('cuda:0')
        self.dtype = torch.half
        # 1. init model
        # self.model, self.tokenizer, self.n_kv_heads, self.head_size, self.n_layers, self.max_seq_len = load_model_tokenizer(model_name= 'gpt2', dtype=self.dtype, device=self.device)
        self.model, self.tokenizer, self.n_kv_heads, self.head_size, self.n_layers, self.max_seq_len = load_model_tokenizer(model_name='meta-llama/Llama-2-7b-hf', dtype=self.dtype, device=self.device)
        print(f'model info: n_kv_heads  {self.n_kv_heads} head_size   {self.head_size} n_layers    {self.n_layers} max_seq_len {self.max_seq_len}')
        print()
        # 2. init kv cache
        self.block_size = 16
        self.num_blocks = 4 * 1024 * 1024 * 1024 // 2 // self.n_layers // self.block_size // self.n_kv_heads // self.head_size 
        self.kv_caches = []
        for _ in range(self.n_layers):
            key_cache = torch.empty(self.num_blocks, self.block_size, self.n_kv_heads, self.head_size, device=self.device, dtype=self.dtype)
            value_cache = torch.empty(self.num_blocks, self.block_size, self.n_kv_heads, self.head_size, device=self.device, dtype=self.dtype)
            self.kv_caches.append(KVCache(key_cache, value_cache))
        self.allocator = BlockAllocator(self.num_blocks)
        print(f'kv cache info block_size {self.block_size} num blocks {self.num_blocks}')
        # 3. capture cuda graph for fast decode
        self.model_runner = self.model
        # self.model_runner = CudaGraphModelRunner(model_runner=self.model, dtype=self.dtype,device=self.device,block_size=self.block_size, vocab_size=self.tokenizer.vocab_size, kv_caches=self.kv_caches, cuda_graph_max_batch_size=64, cuda_graph_max_seq_len=1024)
        
        # 4. batch policy
        self.sequence_id_allocator: int = 0
        self.max_tokens_per_batch : int = 1024
        self.chunk_size           : int = 256
        self.new_queue    : queue.Queue = queue.Queue()
        self.prefill_queue: queue.Queue = queue.Queue()
        self.decode_queue : queue.Queue = queue.Queue()

    def add_request(self, prompt: str) -> int:
        # now it's user responsibility to not pass in prompt length greater than model can forward
        self.sequence_id_allocator += 1
        token_ids = self.tokenizer.encode(prompt)
        sequence = Sequence(
            id = self.sequence_id_allocator, 
            token_ids = token_ids, 
            num_prompt_tokens = len(token_ids), 
            eos_token_id = self.tokenizer.eos_token_id, 
            max_seq_len  = self.max_seq_len
        ) 
        self.new_queue.put(sequence)

        return sequence.id
    
    def continous_batch(self):
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


    @torch.inference_mode()
    def step(self) -> tuple[list[Sequence], list[Sequence]]:
        # 1. batch and allocate memory
        batch_sequences, q_seq_lens = self.continous_batch()
        if len(batch_sequences) == 0:
            return [], []
        # 2. prepare input
        token_ids         : list[int] = []
        positions         : list[int] = []
        q_cu_seq_lens     : list[int] = [0]
        kv_cu_seq_lens    : list[int] = [0]
        block_tables      : list[int] = []
        cu_blocks_lens    : list[int] = [0]
        new_cache_slots   : list[int] = []
        selected_token_ids: list[int] = []
        for sequence, q_seq_len in zip(batch_sequences, q_seq_lens):
            kv_seq_len = sequence.n_kv_cache_tokens + q_seq_len
            for i in range(sequence.n_kv_cache_tokens, kv_seq_len):
                token_ids.append(sequence.token_ids[i])
                positions.append(i)
                new_cache_slots.append(sequence.block_table[i // self.block_size] * self.block_size + i % self.block_size)
            q_cu_seq_lens.append(q_cu_seq_lens[-1] + q_seq_len)
            kv_cu_seq_lens.append(kv_cu_seq_lens[-1] + kv_seq_len)
            block_tables += sequence.block_table
            cu_blocks_lens.append(cu_blocks_lens[-1] + len(sequence.block_table))
            if kv_seq_len >= sequence.num_prompt_tokens:
                selected_token_ids.append(len(token_ids) - 1)

        input_ids = torch.tensor(token_ids, dtype=torch.int, device=self.device)
        position_ids = torch.tensor(positions, dtype=torch.int, device=self.device)
        input_params = InputParameters(
            num_sequences = len(batch_sequences), 
            q_cu_seq_lens = torch.tensor(q_cu_seq_lens, dtype=torch.int, device=self.device),
            kv_cu_seq_lens = torch.tensor(kv_cu_seq_lens, dtype=torch.int, device=self.device), 
            new_cache_slots = torch.tensor(new_cache_slots, dtype=torch.int, device=self.device), 
            block_tables = torch.tensor(block_tables, dtype=torch.int, device=self.device), 
            cu_blocks_lens = torch.tensor(cu_blocks_lens, dtype=torch.int, device=self.device)
        )
        
        # 3. forward
        logits = self.model_runner.forward(input_ids, position_ids, self.kv_caches, input_params)
        for sequence, q_seq_len in zip(batch_sequences, q_seq_lens):
            sequence.n_kv_cache_tokens += q_seq_len
        
        # 4. sample
        if len(selected_token_ids) > 0:
            sample_token_ids = torch.argmax(logits[selected_token_ids, :], dim=-1, keepdim=False).to(torch.device('cpu'))

            i = 0
            for sequence in batch_sequences:
                if sequence.is_decode:
                    token_id = sample_token_ids[i].item()
                    i += 1
                    sequence.token_ids.append(token_id)
            
        # 5. put sequnce batch to queue and free finished
        decode_output: list[Sequence] = []
        finished_output: list[Sequence] = []
        for sequence in batch_sequences:
            if sequence.is_prefill_but_not_new:
                self.prefill_queue.put(sequence)
            elif sequence.is_decode_but_not_finished:
                decode_output.append(sequence)
                self.decode_queue.put(sequence)
            elif sequence.is_finished:
                finished_output.append(sequence)
                self.allocator.free(sequence.block_table)
            else:
                raise Exception
        
        return decode_output, finished_output