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
        print(f'num_blocks: {self.num_blocks}')
        self.kv_caches = []
        for _ in range(self.config.n_layer):
            key_cache = torch.empty(self.num_blocks, self.block_size, self.config.n_head, self.head_size, device=torch.device('cuda:0'), dtype=torch.half)
            value_cache = torch.empty(self.num_blocks, self.block_size, self.config.n_head, self.head_size, device=torch.device('cuda:0'), dtype=torch.half)
            self.kv_caches.append(KVCache(key_cache, value_cache))

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
        
        for i, sequence in enumerate(sequences):
            print(f'sequence{i} {sequence.token_ids}')
        print(f'input_ids          {input_ids}')
        print(f'position_ids       {position_ids}')
        print(f'q_cu_seq_lens      {q_cu_seq_lens}')
        print(f'kv_cu_seq_lens     {kv_cu_seq_lens}')
        print(f'block_tables       {block_tables}')
        print(f'cu_blocks_lens     {cu_blocks_lens}')
        print(f'new_cache_slots    {new_cache_slots}')
        print(f'selected_token_ids {selected_token_ids}')
        
        
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
        for sequence in sequences:
            sequence.n_kv_cache_tokens = len(sequence.token_ids)
        output = self.model(input_ids, position_ids, self.kv_caches, input_params)

        # 3. sample
        logits = output['logits']
        sample_token_ids = torch.argmax(logits[selected_token_ids, :], dim=-1, keepdim=False).to(torch.device('cpu'))
        print(f'sample_token_ids   {sample_token_ids}')
        for i, token_id in enumerate(sample_token_ids):
            print(f'sequence{i} append {token_id}')
            sequences[i].token_ids.append(token_id.item())
        for i, sequence in enumerate(sequences):
            print(f'sequence{i} {sequence.token_ids}')

        print(f'------------------------------------------------------------')