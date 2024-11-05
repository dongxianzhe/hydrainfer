import torch
from torch import nn, Tensor

def compute_default_inv_freq(rotary_dim: int, theta: float) -> Tensor:
    # return [a, b] (rotary_dim / 2, )
    assert rotary_dim % 2 == 0, "rotary_dim must be even"
    slice = torch.arange(0, rotary_dim, 2, dtype=torch.float)
    return 1. / torch.pow(theta, slice / rotary_dim)

class RotaryEmbedding(nn.Module):
    def __init__(self, rotary_dim: int, max_position_embeddings: int, inv_freq: Tensor, interleaved: bool):
        super().__init__()
        # rotary_dim <= head_dim, rotary rotary_dim elements
        # interleaved = True means adjacent two elements rotary as a pair
        self.rotary_dim              = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.inv_freq                = inv_freq # (rotary_dim / 2)
        self.interleaved             = interleaved

        t = torch.arange(self.max_position_embeddings, dtype=torch.float) # (max_position_embeddings)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq) # (max_position_embedding, rotary_dim / 2)
        if self.interleaved:
            # freq [a, b] => emd [a, a, b, b]
            emd = freqs.repeat_interleave(repeats=2, dim=-1)
        else:
            # freq [a, b] => emd [a, b, a, b]
            emd = torch.cat([freqs, freqs], dim=-1)
        # emd (max_position_embedding, rotary_dim)
        cos_sin = torch.cat([emd.cos(), emd.sin()], dim=-1)
        # cos_sin (max_position_embedding, rotary_dim * 2)
        # [ca, ca, cb, cb, sa, sa, sb, sb] or [ca, cb, ca, cb, sa, sb, sa, sb]
        self.register_buffer(name='cos_sin_cache', tensor=cos_sin, persistent=False)
    
    def forward(self, query: Tensor, key: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        dtype = query.dtype
        # query (num_tokens, n_qo_heads, head_dim)
        # key   (num_tokens, n_kv_heads, head_dim)
        # position_ids (num_tokens, )
        query_rotary = query[:, :, : self.rotary_dim]
        query_pass   = query[:, :, self.rotary_dim: ]
        key_rotary = key[:, :, : self.rotary_dim]
        key_pass   = key[:, :, self.rotary_dim: ]

        # cos_sin (num_tokens, rotary_dim * 2)
        cos_sin = torch.nn.functional.embedding(position_ids, self.cos_sin_cache)[:, None, :]
        cos,sin = cos_sin.chunk(chunks=2, dim=-1) # cos(num_tokens, 1, head_dim) sin(num_tokens, 1, head_dim)
        if self.interleaved:
            # rotary_dim = 4
            # cos [ca, ca, cb, cb]
            # sin [sa, sa, sb, sb]
            # query_rotary [0, 1, 2, 3]
            #   [0,   1,  2,  3]
            # * [ca, ca, cb, cb]
            # + [-1,  0, -3,  2]
            # * [sa, sa, sb, sb]
            # counterclock
            query_rotary = (query_rotary * cos) + (self.rotate_every_two(query_rotary) * sin)
            key_rotary = (key_rotary * cos) + (self.rotate_every_two(key_rotary) * sin)
        else:
            # rotary_dim = 4
            # cos [ca, cb, ca, cb]
            # sin [sa, sb, sa, sb]
            # query_rotary [0, 1, 2, 3]
            #   [0,   1,  2,  3]
            # * [ca, cb, ca, cb]
            # + [-2,  -3, 0,  1]
            # * [sa, sb, sa, sb]
            # counterclock
            query_rotary = (query_rotary * cos) + (self.rotate_half(query_rotary) * sin)
            key_rotary = (key_rotary * cos) + (self.rotate_half(key_rotary) * sin)
        query = torch.cat([query_rotary, query_pass], dim=-1)
        key = torch.cat([key_rotary, key_pass], dim=-1)
        return query.to(dtype), key.to(dtype)
        
    def rotate_every_two(self, x: Tensor) -> Tensor:
        # (n_tokens, n_heads, head_dim)
        # [0, 1, 2, 3] => [-1, 0, -3, 2]
        x1 = x[:, :, ::2] # [0, 2]
        x2 = x[:, :, 1::2] # [1, 3]
        x = torch.stack([-x2, x1], dim=-1) # [[-1, 0], [-3, 2]]
        x = x.flatten(start_dim=-2) #  [-1, 0, -3, 2]
        return x
    
    def rotate_half(self, x: Tensor) -> Tensor:
        # (n_tokens, n_heads, head_dim)
        # [0, 1, 2, 3] => [-2, -3, 0, 1]
        x1, x2 = x.chunk(chunks=2, dim=-1) # [0, 1] [2, 3]
        return torch.cat([-x2, x1], dim=-1) # [-2, -3, 0, 1]
