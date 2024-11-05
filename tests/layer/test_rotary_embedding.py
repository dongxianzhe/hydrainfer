import torch
from torch import nn, Tensor
from dxz.layer.rotary_embedding import RotaryEmbedding, compute_default_inv_freq
import pytest
from torch.nn import functional as F

def precompute_freqs_cis(dim, max_position_embeddings, theta):
    range_tensor = torch.arange(0, dim, 2, dtype=torch.float32)
    slice_tensor = range_tensor[:dim // 2]
    freqs = 1.0 / torch.pow(theta, slice_tensor / dim)
    t = torch.arange(max_position_embeddings, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)

# [1, 2, 3, 4, 5, 6] => [1, 3, 5, 2, 4, 6]
def interleaved_to_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.cat([x1, x2], dim=-1)

# [1, 3, 5, 2, 4, 6] -> [1, 2, 3, 4, 5, 6]
def half_to_interleaved(x):
    chunks = x.chunk(2, dim=-1)
    return torch.stack([chunks[0], chunks[1]], dim=-1).flatten(start_dim=-2)

def split_tensor_by_last_dim(x):
    shape = list(x.shape)
    shape[-1] = -1
    shape.append(2)
    return x.reshape(shape)

def apply_rotary_emb(xq, xk, freqs_cis):
    # Convert to complex numbers
    xq_complex = torch.view_as_complex(split_tensor_by_last_dim(xq.to(torch.float)))
    xk_complex = torch.view_as_complex(split_tensor_by_last_dim(xk.to(torch.float)))

    # Reshape for broadcast at n_heads dim
    freqs_cis = freqs_cis.unsqueeze(1)

    # Apply rotation
    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(2)

    # Cast back to original dtype
    xq_out = xq_out.type_as(xq)
    xk_out = xk_out.type_as(xk)

    return xq_out, xk_out

def apply_rotary_emb_ref(query:Tensor, key:Tensor, positions:Tensor, head_dim:int, max_position_embeddings:int, theta:float, interleaved:bool)->tuple[Tensor, Tensor]:
    freqs_cis = precompute_freqs_cis(head_dim, max_position_embeddings, theta)
    selected_freqs_cis = F.embedding(positions, freqs_cis)

    if interleaved:
        return apply_rotary_emb(query, key, selected_freqs_cis) 
    else:
        interleaved_query = half_to_interleaved(query) 
        interleaved_key = half_to_interleaved(key) 
        query_ref, key_ref = apply_rotary_emb(interleaved_query, interleaved_key, selected_freqs_cis)
        query_ref = interleaved_to_half(query_ref) 
        key_ref = interleaved_to_half(key_ref) 
        return query_ref, key_ref

@pytest.mark.parametrize("device", [torch.device('cpu'), torch.device('cuda:0')])
@pytest.mark.parametrize("dtype", [torch.float, torch.half])
@pytest.mark.parametrize("num_tokens", [1, 2, 8, 16])
@pytest.mark.parametrize("n_heads", [32])
@pytest.mark.parametrize("n_kv_heads", [32, 8, 1])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("theta", [100000., 500000.])
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("max_position_embeddings", [4096, 8192])
def test_rotary_correctness(device, dtype, num_tokens, n_heads, n_kv_heads, head_dim, theta, interleaved, max_position_embeddings):
    rotary_embedding = RotaryEmbedding(
        rotary_dim=head_dim, # rotary all elements
        max_position_embeddings=max_position_embeddings,
        inv_freq=compute_default_inv_freq(rotary_dim=head_dim, theta=theta),
        interleaved=interleaved
        )
    rotary_embedding.to(device)
    query = torch.randn(size=(num_tokens, n_heads, head_dim), device=device)
    key = torch.randn(size=(num_tokens, n_kv_heads, head_dim), device=device)
    position_ids = torch.randint(0, max_position_embeddings, size=(num_tokens, ), dtype=torch.int, device=device)
    query_output, key_output = rotary_embedding(query, key, position_ids)

    # ref
    query = query.to(torch.device('cpu'))
    key = key.to(torch.device('cpu'))
    position_ids = position_ids.to(torch.device('cpu'))

    query_ref, key_ref = apply_rotary_emb_ref(
        query = query, 
        key = key, 
        positions = position_ids, 
        head_dim = head_dim, 
        max_position_embeddings = max_position_embeddings, 
        theta = theta, 
        interleaved = interleaved
    )

    query_output = query_output.to(torch.device('cpu'))
    key_output = key_output.to(torch.device('cpu'))
    assert torch.allclose(query_ref, query_output, rtol=1e-3, atol=1e-5) 
    assert torch.allclose(key_ref, key_output, rtol=1e-3, atol=1e-5) 

if __name__ == "__main__":
    pytest.main([__file__])