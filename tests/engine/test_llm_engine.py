from dxz.engine.llm_engine import LLMEngine
from dxz.request.sequence import Sequence
from dxz.memory.block_allocator import BlockAllocator
import torch
from transformers import GPT2LMHeadModel as GPT2LMHeadModelRef
from transformers import GPT2Tokenizer

def generate_ref(prompt: str, max_tokens=50) -> str:
    device=torch.device('cuda:0')
    dtype =torch.half

    model = GPT2LMHeadModelRef.from_pretrained("gpt2")
    model.to(dtype)
    model.to(device)
    token_ids = tokenizer.encode(prompt) # token list # (n_tokens, )
    input_ids = torch.tensor(token_ids, dtype=torch.int, device=device)
    for _ in range(max_tokens):
        position_ids = torch.arange(input_ids.shape[-1], device=device, dtype=torch.int) # (n_tokens, )
        logits = model(input_ids=input_ids, position_ids=position_ids)['logits'] # (n_tokens, vocab_size)
        next_token_id = torch.argmax(logits[-1, :], dim=-1, keepdim=True) # (1, )

        if(next_token_id.item() == tokenizer.eos_token_id):
            break

        input_ids = torch.cat((input_ids, next_token_id), dim=0) # (n_tokens + 1, )
    
    return input_ids

prompts = [
    "Okay, let's rewrite this. We need some conversation between Jack and his mother immediately after - perhaps over dinner back at their house? She is concerned, but not doubting him too much, and he has to convince her to let him come back to the museum?", 
    "what's the weather like today?",
    "who are you ?",
    "hello world."
    ]

if __name__ == '__main__':
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    max_tokens = 50

    # engine
    llm_engine = LLMEngine()
    unfinished_sequences: list[Sequence] = []
    for i, prompt in enumerate(prompts):
        token_ids = tokenizer.encode(prompt)
        sequence = Sequence(
            id = i, 
            token_ids = token_ids, 
            num_prompt_tokens = len(token_ids)
        ) 
        unfinished_sequences.append(sequence)

    finished_sequences: list[Sequence] = []
    while len(unfinished_sequences) > 0:
        f, unfinished_sequences = llm_engine.execute_model(unfinished_sequences)
        finished_sequences+= f

    for i, sequence in enumerate(finished_sequences):
        o = tokenizer.decode(sequence.token_ids)
        o_ref = tokenizer.decode(generate_ref(prompts[sequence.id]))
        print('============================================================')
        print(f'my : {o}')
        print('------------------------------------------------------------')
        print(f'ref: {o_ref}')
        print('============================================================')
        # assert o == o_ref