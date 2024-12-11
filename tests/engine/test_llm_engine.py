from dxz.engine.llm_engine import LLMEngine, Sequence
from dxz.memory.block_allocator import BlockAllocator
import torch
from transformers import GPT2LMHeadModel as GPT2LMHeadModelRef
from transformers import GPT2Tokenizer

def generate_ref(prompt: str, max_tokens=50) -> str:
    device=torch.device('cuda:0')
    dtype =torch.half

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
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

def test_forward():
    # engine
    llm_engine = LLMEngine()
    tokenizer = llm_engine.tokenizer
    for prompt in prompts:
        seq_id = llm_engine.add_request(prompt)
    
    sequences: list[Sequence] = []
    for i in range(100): # step 100 to ensure finish all sequences
        _, finished = llm_engine.step()
        sequences += finished

    for sequence in sequences: 
        o = tokenizer.decode(sequence.token_ids)
        o_ref = tokenizer.decode(generate_ref(prompts[sequence.id - 1]))
        print('============================================================')
        print(f'my : {o}')
        print('------------------------------------------------------------')
        print(f'ref: {o_ref}')
        print('============================================================')
        # assert o == o_ref

def test_input():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    all = []
    for prompt in prompts:
        token_ids = tokenizer.encode(prompt)
        print(f'len token_ids {len(token_ids)}')
        all += token_ids
    print(f'total token_ids {len(all)}')


def test_batch_policy():
    llm_engine = LLMEngine()
    tokenizer = llm_engine.tokenizer
    num_requests = 10000
    for i in range(num_requests):
        seq_id = llm_engine.add_request(prompts[0])
    sequences: list[Sequence] = []
    for i in range(10000): # step 100 to ensure finish all sequences
        _, finished = llm_engine.step()
        sequences += finished

    assert len(sequences) == num_requests

if __name__ == '__main__':
    test_forward()
    # test_input()
    # test_batch_policy()