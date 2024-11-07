import torch
from dxz.model.parameters import InputParameters
from dxz.model.model_loader import load_model_tokenizer
from dxz.memory.kv_cache import KVCache
from dxz.request.sequence import Sequence
from dxz.memory.block_allocator import BlockAllocator
from dxz.model_runner.cuda_graph_model_runner import CudaGraphModelRunner
from dxz.engine.continuous_batch import ContinuousBatch

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
        self.kv_caches = [KVCache(self.num_blocks, self.block_size, self.n_kv_heads, self.head_size, dtype=self.dtype, device=self.device) for _ in range(self.n_layers)]
        self.allocator = BlockAllocator(self.num_blocks)
        print(f'kv cache info block_size {self.block_size} num blocks {self.num_blocks}')
        # 3. capture cuda graph for fast decode
        self.model_runner = self.model
        # self.model_runner = CudaGraphModelRunner(model_runner=self.model, dtype=self.dtype,device=self.device,block_size=self.block_size, vocab_size=self.tokenizer.vocab_size, kv_caches=self.kv_caches, cuda_graph_max_batch_size=1, cuda_graph_max_seq_len=self.max_seq_len)
        
        # 4. batch policy
        self.sequence_id_allocator: int = 0
        self.batch_policy = ContinuousBatch(max_tokens_per_batch = 1024, chunk_size = 256, allocator=self.allocator, block_size=self.block_size)

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
        self.batch_policy.add_new_sequence(sequence)

        return sequence.id

    @torch.inference_mode()
    def step(self) -> tuple[list[Sequence], list[Sequence]]:
        # 1. batch and allocate memory
        batch_sequences, q_seq_lens = self.batch_policy.batch()
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
        kv_seq_lens       : list[int] = []
        for sequence, q_seq_len in zip(batch_sequences, q_seq_lens):
            kv_seq_len = sequence.n_kv_cache_tokens + q_seq_len
            kv_seq_lens.append(kv_seq_len)
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
            cu_blocks_lens = torch.tensor(cu_blocks_lens, dtype=torch.int, device=self.device), 
            q_max_seq_len = max(q_seq_lens), 
            kv_max_seq_len = max(kv_seq_lens)
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
                self.batch_policy.add_prefill_sequence(sequence)
            elif sequence.is_decode_but_not_finished:
                decode_output.append(sequence)
                self.batch_policy.add_decode_sequence(sequence)
            elif sequence.is_finished:
                finished_output.append(sequence)
                self.allocator.free(sequence.block_table)
            else:
                raise Exception
        
        return decode_output, finished_output