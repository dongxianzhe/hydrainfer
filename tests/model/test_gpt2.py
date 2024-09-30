import pytest
import torch
from torch import nn

def count_parameters(model: nn.Module):
    total = 0
    param_required_grad = 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            param_required_grad += param.numel()
    return total, param_required_grad

def test_GPT2Model():
    from transformers import GPT2Config
    from dxz.model import GPT2Model
    from transformers import GPT2Model as GPT2ModelRef
    config = GPT2Config()
    model = GPT2Model(config)
    model_ref = GPT2ModelRef.from_pretrained('gpt2')

    count = count_parameters(model)
    count_ref = count_parameters(model)
    assert count == count_ref, f"params count wrong {count} != {count_ref}"

    model.eval()
    model_ref.eval()
    state_dict = model_ref.state_dict()
    model.load_weights(state_dict)

    s1 = model.state_dict()
    s2 = model_ref.state_dict()
    for name, weight in s1.items():
        if any(needle in name for needle in ["c_attn.weight", "c_proj.weight", "c_fc.weight"]):
            weight_ref = s2[name].t()
        else:
            weight_ref = s2[name]

        assert torch.allclose(weight, weight_ref, rtol=1e-3, atol=1e-3), f"layer {name} not equal"
        
    input_ids = torch.randint(0, config.vocab_size, (config.max_position_embeddings,))
    position_ids = torch.arange(config.max_position_embeddings)

    output = model(input_ids=input_ids, position_ids=position_ids)
    output_ref = model_ref(input_ids=input_ids, position_ids=position_ids, output_hidden_states = True)
    for i, (layer_i, layer_i_ref) in enumerate(zip(output['hidden_states'], output_ref['hidden_states'])):
        assert torch.allclose(layer_i.reshape(-1), layer_i_ref.reshape(-1), rtol=1e-3, atol=1e-3), {f'layer {i} input hidden_states is wrong'}

    assert torch.allclose(output['last_hidden_state'], output_ref['last_hidden_state'], rtol=1e-3, atol=1e-3), "outout not equal"
    print('pass')

def test_gpt2lmhead_output():
    from transformers import GPT2Config
    from dxz.model import GPT2LMHeadModel
    from transformers import GPT2LMHeadModel as GPT2LMHeadModelRef
    config = GPT2Config()
    model = GPT2LMHeadModel(config)
    model_ref = GPT2LMHeadModelRef.from_pretrained('gpt2')
    model.eval()
    model_ref.eval()

    state_dict = model_ref.state_dict()
    model.load_weights(state_dict)


    input_ids = torch.randint(0, config.vocab_size, (config.max_position_embeddings,))
    position_ids = torch.arange(config.max_position_embeddings)

    output = model(input_ids=input_ids, position_ids=position_ids)
    output_ref = model_ref(input_ids=input_ids, position_ids=position_ids, output_hidden_states = True)
    assert torch.allclose(output['logits'], output_ref['logits'], rtol=1e-3, atol=1e-3), "outout not equal"
    print('pass')

if __name__ == '__main__':
    pytest.main()