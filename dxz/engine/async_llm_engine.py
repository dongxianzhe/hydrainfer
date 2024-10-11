import torch
from transformers import GPT2Config, GPT2Tokenizer
from transformers import GPT2LMHeadModel as GPT2LMHeadModelRef
from dxz.model.gpt2 import GPT2LMHeadModel
from dxz.model.gpt2 import InputParameters

class AsyncLLMEngine:
    def __init__(self) -> None:
        # tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # model
        self.config = GPT2Config()
        self.model = GPT2LMHeadModel(self.config)
        self.model.load_state_dict(GPT2LMHeadModelRef.from_pretrained('gpt2').state_dict())
        self.model.to(device=torch.device('cuda:0'))

    async def generate_page_attention(self, prompts: list[str]) -> list[str]:
        max_tokens = 50
        results = []
        from dxz.memory.kv_cache import KVCache

        for prompt in prompts:
            # 1. create input
            num_blocks = 10
            block_size = 16
            hidden_size = self.config.n_embd // self.config.n_head
            kv_caches = []
            for _ in range(self.config.n_layer):
                key_cache = torch.empty(num_blocks, block_size, self.config.n_head, hidden_size, device=torch.device('cuda:0'))
                value_cache = torch.empty(num_blocks, block_size, self.config.n_head, hidden_size, device=torch.device('cuda:0'))
                kv_caches.append(KVCache(key_cache, value_cache))

            input_ids = self.tokenizer.encode(prompt, return_tensors='pt')[0].to(torch.device('cuda:0'))
            prompt_length = input_ids.numel()
            position_ids = torch.arange(prompt_length, device=torch.device('cuda:0')) # (n_tokens, )

            input_params = InputParameters(
                num_sequences = 1, 
                q_cu_seq_lens = torch.tensor([0, prompt_length], dtype=torch.int), 
                kv_cu_seq_lens = torch.tensor([0, prompt_length], dtype=torch.int), 
                new_cache_slots = torch.arange(prompt_length, dtype=torch.int), 
                block_tables = torch.arange(num_blocks), 
                cu_blocks_lens = torch.tensor([0, num_blocks], dtype=torch.int)
            )

            output = self.tokenizer.encode(prompt)

            for _ in range(max_tokens):
                # 3. forward
                logits = self.model(input_ids, position_ids, kv_caches, input_params)['logits'] # (n_tokens, vocab_size)
                # 4. sample
                next_token_logits = logits[-1] # (vocab_size, )
                next_token_id = torch.argmax(next_token_logits, dim=-1).item()
                output.append(next_token_id)
                # 5. create next input
                input_ids = torch.tensor([next_token_id], dtype=torch.long, device=torch.device('cuda:0'))
                position_ids = torch.tensor([len(output) - 1], dtype=torch.long, device=torch.device('cuda:0'))
                input_params = InputParameters(
                    num_sequences = 1, 
                    q_cu_seq_lens = torch.tensor([0, 1], dtype=torch.int), 
                    kv_cu_seq_lens = torch.tensor([0, len(output)], dtype=torch.int), 
                    new_cache_slots = torch.tensor([len(output) - 1], dtype=torch.int), 
                    block_tables = torch.arange(num_blocks), 
                    cu_blocks_lens = torch.tensor([0, num_blocks], dtype=torch.int)
                )
            results.append(self.tokenizer.decode(output))
        return results