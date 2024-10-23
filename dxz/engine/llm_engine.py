import torch
from transformers import GPT2Config, GPT2Tokenizer
from transformers import GPT2LMHeadModel as GPT2LMHeadModelRef
from dxz.model.gpt2 import GPT2LMHeadModel
from dxz.model.gpt2 import InputParameters
from dxz.memory.kv_cache import KVCache
from dxz.request.sequence import Sequence
from dxz.memory.block_allocator import BlockAllocator

class LLMEngine:
    def __init__(self):
        self.device = torch.device('cuda:0')
        # 1. init model
        self.config = GPT2Config()
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel(self.config)
        self.model.load_state_dict(GPT2LMHeadModelRef.from_pretrained('gpt2').state_dict())
        self.model.half()
        self.model.to(device=self.device)
        # 2. init kv cache
        self.block_size = 16
        self.head_size = self.config.n_embd // self.config.n_head
        self.num_blocks = 4 * 1024 * 1024 * 1024 // 2 // self.config.n_embd // self.block_size // self.config.n_layer
        self.kv_caches = []
        for _ in range(self.config.n_layer):
            key_cache = torch.empty(self.num_blocks, self.block_size, self.config.n_head, self.head_size, device=self.device, dtype=torch.half)
            value_cache = torch.empty(self.num_blocks, self.block_size, self.config.n_head, self.head_size, device=self.device, dtype=torch.half)
            self.kv_caches.append(KVCache(key_cache, value_cache))
        self.allocator = BlockAllocator(self.num_blocks)
        # 3. capture cuda graph for fast decode
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {} # batch_size -> cuda graph
        cuda_graph_max_batch_size = 64
        cuda_graph_max_seq_len = 1024
        cuda_graph_max_block_len_per_seq = (cuda_graph_max_batch_size + self.block_size - 1) // self.block_size
        cuda_graph_max_block_len = cuda_graph_max_batch_size * cuda_graph_max_block_len_per_seq
        self.num_blocks_per_seq = num_blocks_per_seq = 5
        self.static_input_ids = torch.empty(cuda_graph_max_batch_size, dtype=torch.int, device=self.device)
        self.static_position_ids = torch.empty(cuda_graph_max_batch_size, dtype=torch.int, device=self.device)
        self.static_q_cu_seq_lens = torch.empty(cuda_graph_max_batch_size + 1, dtype=torch.int, device=self.device)
        self.static_kv_cu_seq_lens = torch.empty(cuda_graph_max_batch_size + 1, dtype=torch.int, device=self.device)
        self.static_new_cache_slots = torch.empty(cuda_graph_max_batch_size, dtype=torch.int, device=self.device)
        self.static_block_tables = torch.empty(cuda_graph_max_block_len, dtype=torch.int, device=self.device)
        self.static_cu_blocks_lens = torch.empty(cuda_graph_max_batch_size + 1, dtype=torch.int, device=self.device)
        self.static_logits = torch.empty((cuda_graph_max_batch_size, self.config.vocab_size), dtype=torch.half, device=self.device)
        for batch_size in range(1, cuda_graph_max_batch_size, 1):
            print(f'cuda capture batch_size {batch_size}')
            input_ids = self.static_input_ids[:batch_size]
            position_ids = self.static_position_ids[:batch_size]
            input_params = InputParameters(
                num_sequences = batch_size, 
                q_cu_seq_lens = self.static_q_cu_seq_lens[:batch_size+1], 
                kv_cu_seq_lens = self.static_kv_cu_seq_lens[:batch_size+1], 
                new_cache_slots = self.static_new_cache_slots[:batch_size], 
                block_tables = self.static_block_tables, 
                cu_blocks_lens = self.static_cu_blocks_lens[:batch_size+1]
            )
            # Run the model a few times without capturing the graph.
            # This is to make sure that the captured graph does not include the
            # kernel launches for initial benchmarking (e.g., Triton autotune).
            # Note one iteration is not enough for torch.jit.script
            self.model(input_ids, position_ids, self.kv_caches, input_params)
            self.model(input_ids, position_ids, self.kv_caches, input_params)
            torch.cuda.synchronize()

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self.static_logits[:batch_size] = self.model(input_ids, position_ids, self.kv_caches, input_params)
            torch.cuda.synchronize()
            self.graphs[batch_size] = g

    @torch.inference_mode()
    def execute_model(self, sequences: list[Sequence]):
        # 1. allocate block
        for sequence in sequences:
            while len(sequence.block_table) * self.block_size < len(sequence.token_ids):
                id = self.allocator.allocate(1)[0]
                sequence.block_table.append(id)
        # 2. prepare model input
        num_sequences = len(sequences)
        input_ids: list[int] = []
        position_ids: list[int] = []
        q_cu_seq_lens: list[int] = [0]
        kv_cu_seq_lens: list[int] = [0]
        block_tables: list[int] = []
        cu_blocks_lens: list[int] = [0]
        new_cache_slots: list[int] = []
        selected_token_ids: list[int] = []
        for sequence in sequences:
            for i in range(sequence.n_kv_cache_tokens, len(sequence.token_ids)):
                input_ids.append(sequence.token_ids[i])
                position_ids.append(i)
                new_cache_slots.append(sequence.block_table[i // self.block_size] * self.block_size + i % self.block_size)
            q_cu_seq_lens.append(q_cu_seq_lens[-1] + len(sequence.token_ids) - sequence.n_kv_cache_tokens)
            kv_cu_seq_lens.append(kv_cu_seq_lens[-1] + len(sequence.token_ids))
            block_tables += sequence.block_table
            cu_blocks_lens.append(cu_blocks_lens[-1] + len(sequence.block_table))
            selected_token_ids.append(len(input_ids) - 1)
        
        input_ids = torch.tensor(input_ids, dtype=torch.int, device=self.device)
        position_ids = torch.tensor(position_ids, dtype=torch.int, device=self.device)
        input_params = InputParameters(
            num_sequences = num_sequences, 
            q_cu_seq_lens = torch.tensor(q_cu_seq_lens, dtype=torch.int, device=self.device),
            kv_cu_seq_lens = torch.tensor(kv_cu_seq_lens, dtype=torch.int, device=self.device), 
            new_cache_slots = torch.tensor(new_cache_slots, dtype=torch.int, device=self.device), 
            block_tables = torch.tensor(block_tables, dtype=torch.int, device=self.device), 
            cu_blocks_lens = torch.tensor(cu_blocks_lens, dtype=torch.int, device=self.device)
        )
        # 3. forward
        all_sequence_is_decode: bool = all(sequence.n_kv_cache_tokens != 0 for sequence in sequences)
        for sequence in sequences:
            sequence.n_kv_cache_tokens = len(sequence.token_ids)
        if all_sequence_is_decode and num_sequences in self.graphs:
            g = self.graphs[num_sequences]
            self.static_input_ids[: num_sequences].copy_(input_ids)
            self.static_position_ids[: num_sequences].copy_(position_ids)
            self.static_q_cu_seq_lens[: num_sequences + 1].copy_(input_params.q_cu_seq_lens)
            self.static_kv_cu_seq_lens[: num_sequences + 1].copy_(input_params.kv_cu_seq_lens)
            self.static_new_cache_slots[: num_sequences].copy_(input_params.new_cache_slots)
            self.static_block_tables[: input_params.block_tables.shape[-1]].copy_(input_params.block_tables)
            self.static_cu_blocks_lens[: num_sequences + 1].copy_(input_params.cu_blocks_lens)
            g.replay()
            logits = self.static_logits[: num_sequences, :]
        else:
            logits = self.model(input_ids, position_ids, self.kv_caches, input_params)

        # 4. sample
        sample_token_ids = torch.argmax(logits[selected_token_ids, :], dim=-1, keepdim=False).to(torch.device('cpu'))

        # 5. check finish
        for i, (sequence, token_id) in enumerate(zip(sequences, sample_token_ids)):
            if token_id == self.tokenizer.eos_token_id:
                sequence.finished = True
                continue
            sequence.token_ids.append(token_id.item())
            if len(sequence.token_ids) - sequence.num_prompt_tokens == sequence.max_tokens:
                sequence.finished = True
                continue
        # 6. free blocks
        for sequence in sequences:
            if sequence.finished:
                self.allocator.free(sequence.block_table)
        
        finished_sequences = []
        unfinished_sequences = []
        for sequence in sequences:
            if sequence.finished:
                finished_sequences.append(sequence)
            else:
                unfinished_sequences.append(sequence)
        return finished_sequences, unfinished_sequences