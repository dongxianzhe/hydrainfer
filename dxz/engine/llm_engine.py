import torch
from transformers import GPT2Config, GPT2Tokenizer
from transformers import GPT2LMHeadModel as GPT2LMHeadModelRef
from dxz.model.gpt2 import GPT2LMHeadModel

class LLMEngine:
    def __init__(self) -> None:
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel(GPT2Config())
        self.model.load_state_dict(GPT2LMHeadModelRef.from_pretrained('gpt2').state_dict())

    def generate(self, prompts: list[str]) -> list[str]:
        results = []
        for prompt in prompts:
            encoded_input = self.tokenizer.encode(prompt, return_tensors='pt')[0] # token list # (n_tokens, )
            for _ in range(50):
                position_ids = torch.arange(encoded_input.shape[-1]) # (n_tokens, )
                logits = self.model(input_ids=encoded_input, position_ids=position_ids)['logits'] # (n_tokens, vocab_size)
                next_token_logits = logits[-1, :] # (vocab_size, )
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True) # (1, )

                if(next_token_id == self.tokenizer.eos_token_id):
                    break

                encoded_input = torch.cat((encoded_input, next_token_id), dim=0) # (n_tokens + 1, )

            decoded_output = self.tokenizer.decode(encoded_input)
            results.append(decoded_output)
        
        return results

if __name__ == '__main__':
    llm_engine = LLMEngine()
    results = llm_engine.generate(["Hello"])
    for result in results:
        print(result)