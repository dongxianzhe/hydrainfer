import torch
from transformers import GPT2Config, GPT2Tokenizer
from transformers import GPT2LMHeadModel as GPT2LMHeadModelRef
from dxz.model.gpt2 import GPT2LMHeadModel
from dxz.model.gpt2 import ModelForwardParameters

class AsyncLLMEngine:
    def __init__(self) -> None:
        # tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # model
        self.config = GPT2Config()
        self.model = GPT2LMHeadModel(self.config)
        self.model.load_state_dict(GPT2LMHeadModelRef.from_pretrained('gpt2').state_dict())
        self.model.to(device=torch.device('cuda:0'))

    async def generate(self, prompts: list[str]) -> list[str]:
        max_tokens = 50

        results = []
        for prompt in prompts:
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt')[0] # token list # (n_tokens, )
            output_ids: list[int] = []

            total_tokens = input_ids.shape[-1] # when each loop begin, total_tokens is kv_seq_len = cache_length + input_ids's length
            input_ids = input_ids.to(device=torch.device('cuda:0'))
            position_ids = torch.arange(total_tokens, device=torch.device('cuda:0')) # (n_tokens, )
            n_head, n_embd, n_head, n_layer =self.config.n_head, self.config.n_embd, self.config.n_head, self.config.n_layer
            kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = [(
                torch.empty(size=(input_ids.shape[0] + max_tokens, n_head, n_embd // n_head), device=torch.device('cuda:0')),
                torch.empty(size=(input_ids.shape[0] + max_tokens, n_head, n_embd // n_head), device=torch.device('cuda:0')))
                for _ in range(n_layer)
            ]
            model_forward_parameters = ModelForwardParameters(cache_length=0)
            for _ in range(max_tokens):
                logits = self.model(input_ids=input_ids, position_ids=position_ids, kv_caches=kv_caches, model_forward_parameters=model_forward_parameters)['logits'] # (n_tokens, vocab_size)
                next_token_logits = logits[-1, :] # (vocab_size, )
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True) # (1, )
                output_ids.append(next_token_id.item())

                if(next_token_id == self.tokenizer.eos_token_id):
                    break

                total_tokens += 1
                input_ids = next_token_id # (n_tokens=1, )
                position_ids  = torch.tensor(total_tokens - 1, dtype=torch.int, device=torch.device('cuda:0'))
                model_forward_parameters = ModelForwardParameters(cache_length=total_tokens - 1)

            results.append(prompt + self.tokenizer.decode(output_ids))
        
        return results