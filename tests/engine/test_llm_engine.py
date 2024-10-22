from dxz.engine.llm_engine import LLMEngine
from dxz.request.sequence import Sequence
from dxz.memory.block_allocator import BlockAllocator
import torch
from transformers import GPT2LMHeadModel as GPT2LMHeadModelRef
from transformers import GPT2Tokenizer

def generate_ref(prompt: str, max_tokens=50) -> str:
    device=torch.device('cuda:0')
    model = GPT2LMHeadModelRef.from_pretrained("gpt2")
    model.to(device)
    input_ids = tokenizer.encode(prompt, return_tensors='pt')[0] # token list # (n_tokens, )
    for _ in range(max_tokens):
        position_ids = torch.arange(input_ids.shape[-1]) # (n_tokens, )
        input_ids    = input_ids.to(device).to(torch.int)
        position_ids = position_ids.to(device).to(torch.int)
        logits = model(input_ids=input_ids, position_ids=position_ids)['logits'] # (n_tokens, vocab_size)
        next_token_id = torch.argmax(logits[-1, :], dim=-1, keepdim=True) # (1, )

        # if(next_token_id == tokenizer.eos_token_id):
        #     break

        input_ids = torch.cat((input_ids, next_token_id), dim=0) # (n_tokens + 1, )
    
    return input_ids
    # return tokenizer.decode(input_ids)


if __name__ == '__main__':
    # prompts = ["Who are you?", "Hello", "what's the weather like today?"]
    # prompts = ["Who", "Hello"]
    prompts = ["Hello", "World", "world", "who"]
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    max_tokens = 50

    # engine
    llm_engine = LLMEngine()
    allocator = BlockAllocator(100)
    allocator.allocate(1)
    sequences: list[Sequence] = []
    for prompt in prompts:
        sequence = Sequence()
        sequence.token_ids = tokenizer.encode(prompt)
        sequence.n_kv_cache_tokens = 0
        sequence.block_table = allocator.allocate(n_blocks=5)
        sequences.append(sequence)

    for i in range(max_tokens):
        llm_engine.execute_model(sequences)

    for i, sequence in enumerate(sequences):
        o = tokenizer.decode(sequence.token_ids)
        o_ref = tokenizer.decode(generate_ref(prompts[i]))
        print(f'my : {o}')
        print(f'ref: {o_ref}')
        print('------------------------------------------------------------')
        assert o == o_ref
    # Hello, I'm sorry, but I'm not sure if you're aware of this. I'm not sure if you're aware of this. I'm not sure if you're aware of this. I'm not sure if you're aware of this.