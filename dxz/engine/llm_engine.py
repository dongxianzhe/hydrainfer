import torch
from transformers import GPT2Config, GPT2Tokenizer
from transformers import GPT2LMHeadModel as GPT2LMHeadModelRef
from dxz.model.gpt2 import GPT2LMHeadModel
from dxz.model.gpt2 import InputParameters
from dxz.memory.kv_cache import KVCache
from dxz.request.sequence import Sequence

class LLMEngine:
    def __init__(self):
        self.device = torch.device('cuda:0')
        # 1. init model
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.config = GPT2Config()
        self.model = GPT2LMHeadModel(self.config)
        self.model.load_state_dict(GPT2LMHeadModelRef.from_pretrained('gpt2').state_dict())
        self.model.half()
        self.model.to(device=torch.device('cuda:0'))
        # 2. init kv cache
        self.block_size = 16
        self.head_size = self.config.n_embd // self.config.n_head
        self.num_blocks = 4 * 1024 * 1024 * 1024 // 2 // self.config.n_embd // self.block_size // self.config.n_layer
        self.kv_caches = []
        for _ in range(self.config.n_layer):
            key_cache = torch.empty(self.num_blocks, self.block_size, self.config.n_head, self.head_size, device=torch.device('cuda:0'), dtype=torch.half)
            value_cache = torch.empty(self.num_blocks, self.block_size, self.config.n_head, self.head_size, device=torch.device('cuda:0'), dtype=torch.half)
            self.kv_caches.append(KVCache(key_cache, value_cache))
        # 3. capture cuda graph
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        max_num_sequences = 100
        self.num_blocks_per_seq = num_blocks_per_seq = 5
        self.static_input_ids = torch.empty(max_num_sequences, dtype=torch.int, device=self.device)
        self.static_position_ids = torch.empty(max_num_sequences, dtype=torch.int, device=self.device)
        self.static_q_cu_seq_lens = torch.empty(max_num_sequences + 1, dtype=torch.int, device=self.device)
        self.static_kv_cu_seq_lens = torch.empty(max_num_sequences + 1, dtype=torch.int, device=self.device)
        self.static_new_cache_slots = torch.empty(max_num_sequences, dtype=torch.int, device=self.device)
        self.static_block_tables = torch.empty(num_blocks_per_seq * max_num_sequences, dtype=torch.int, device=self.device)
        self.static_cu_blocks_lens = torch.empty(max_num_sequences + 1, dtype=torch.int, device=self.device)
        self.static_logits = torch.empty((max_num_sequences, self.config.vocab_size), dtype=torch.half, device=self.device)
        for num_sequences in range(1, max_num_sequences, 1):
            print(f'num_sequences {num_sequences}')
            input_ids = self.static_input_ids[:num_sequences]
            position_ids = self.static_position_ids[:num_sequences]
            input_params = InputParameters(
                num_sequences = num_sequences, 
                q_cu_seq_lens = self.static_q_cu_seq_lens[:num_sequences+1], 
                kv_cu_seq_lens = self.static_kv_cu_seq_lens[:num_sequences+1], 
                new_cache_slots = self.static_new_cache_slots[:num_sequences], 
                block_tables = self.static_block_tables[:num_blocks_per_seq * num_sequences], 
                cu_blocks_lens = self.static_cu_blocks_lens[:num_sequences+1]
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
                self.static_logits[:num_sequences] = self.model(input_ids, position_ids, self.kv_caches, input_params)
            torch.cuda.synchronize()
            self.graphs[num_sequences] = g

    @torch.inference_mode()
    def execute_model(self, sequences: list[Sequence]):
        # 1. prepare model input
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
        # 2. forward
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
            self.static_block_tables[: self.num_blocks_per_seq * num_sequences].copy_(input_params.block_tables)
            self.static_cu_blocks_lens[: num_sequences + 1].copy_(input_params.cu_blocks_lens)
            g.replay()
            logits = self.static_logits[: num_sequences, :]
        else:
            logits = self.model(input_ids, position_ids, self.kv_caches, input_params)

        # 3. sample
        sample_token_ids = torch.argmax(logits[selected_token_ids, :], dim=-1, keepdim=False).to(torch.device('cpu'))
        for i, token_id in enumerate(sample_token_ids):
            sequences[i].token_ids.append(token_id.item())