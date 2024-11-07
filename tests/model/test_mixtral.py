import torch
from torch import nn, Tensor
from transformers import MixtralConfig
from transformers import MixtralForCausalLM as MixtralForCausalLMRef
from torch.nn import functional
from typing import Optional
from dxz.memory.kv_cache import KVCache
from dxz.model.mixtral import MixtralForCausalLM
from dxz.model.parameters import InputParameters
from dxz.layer.rotary_embedding import compute_default_inv_freq, RotaryEmbedding
from dxz.utils.model_utils import count_parameters, print_parameters

torch.set_grad_enabled(False)

config = MixtralConfig(num_hidden_layers=1, hidden_size=128, intermediate_size=512)
model: MixtralForCausalLM = MixtralForCausalLM(config)
model_ref: MixtralForCausalLMRef = MixtralForCausalLMRef(config)

count = count_parameters(model)
count_ref = count_parameters(model_ref)

assert count == count_ref, f"{count} {count_ref}"

state_dict_ref = model_ref.state_dict()
model.load_state_dict(model_ref.state_dict())


for name, weight in model.named_parameters():
    assert torch.allclose(weight, state_dict_ref[name], rtol=1e-3, atol=1e-5)


# input paramters
n_tokens = 10

input_ids = torch.randint(0, config.vocab_size, size=(n_tokens, ), dtype=torch.int)
position_ids = torch.arange(n_tokens, dtype=torch.int)
num_block = 100
block_size = 16
head_dim = config.hidden_size // config.num_attention_heads
kv_caches = [KVCache(num_block, block_size, config.num_key_value_heads, head_dim, dtype=torch.float, device=torch.device('cpu')) for _ in range(config.num_hidden_layers)]

input_params = InputParameters(
    num_sequences = 1, 
    q_cu_seq_lens = torch.tensor([0, n_tokens], dtype=torch.int), 
    kv_cu_seq_lens = torch.tensor([0, n_tokens], dtype=torch.int), 
    new_cache_slots = torch.arange(n_tokens, dtype=torch.int), 
    block_tables = torch.arange(num_block, dtype=torch.int), 
    cu_blocks_lens = torch.tensor([0, num_block], dtype=torch.int)
)

output = model(input_ids, position_ids, kv_caches, input_params)
assert output['logits'].shape == (n_tokens, config.vocab_size)

output_ref = model_ref(input_ids=input_ids[None, :], position_ids=position_ids[None, :], output_hidden_states=True)
assert output_ref['logits'].shape == (1, n_tokens, config.vocab_size)
assert len(output['hidden_states']) == len(output_ref['hidden_states'])

for i, (h, h_ref) in enumerate(zip(output['hidden_states'], output_ref['hidden_states'])):
    assert torch.allclose(h, h_ref, rtol=1e-3, atol=1e-5), f'input of layer {i} is different'

assert torch.allclose(output['logits'].view(-1), output_ref['logits'].view(-1), rtol=1e-3, atol=1e-5)

print('pass')
