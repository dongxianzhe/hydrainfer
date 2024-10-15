import asyncio
import torch
from transformers import GPT2LMHeadModel as GPT2LMHeadModelRef
from transformers import GPT2Tokenizer
from dxz.engine.async_llm_engine import AsyncLLMEngine
from dxz.request.Sequence import Sequence

if __name__ == '__main__':
    prompt = "Hello"
    # my model
    llm_engine = AsyncLLMEngine()
    results: list[str] = asyncio.run(llm_engine.generate_page_attention([prompt]))
    result: str = results[0]

    # ref
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModelRef.from_pretrained("gpt2")

    encoded_input = tokenizer.encode(prompt, return_tensors='pt')[0] # token list # (n_tokens, )
    for _ in range(50):
        position_ids = torch.arange(encoded_input.shape[-1]) # (n_tokens, )
        logits = model(input_ids=encoded_input, position_ids=position_ids)['logits'] # (n_tokens, vocab_size)
        next_token_logits = logits[-1, :] # (vocab_size, )
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True) # (1, )

        if(next_token_id == tokenizer.eos_token_id):
            break

        encoded_input = torch.cat((encoded_input, next_token_id), dim=0) # (n_tokens + 1, )

    decoded_output = tokenizer.decode(encoded_input)

    print('------------my engine------------------')
    print(result)
    print('---------------ref outout---------------')
    print(decoded_output)
    assert result == decoded_output