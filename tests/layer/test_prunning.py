import torch
from hydrainfer.layer.token_prunning import FocalPrunning

def test_focal_prunning():
    model = FocalPrunning(2 / 3, 'rank')
    batch_size = 2
    n_tokens = 3
    hidden_size = 1024
    n_heads = 2
    dtype = torch.half
    device = torch.device('cuda:0')
    tokens = torch.randn(size=(batch_size, n_tokens, hidden_size), dtype=dtype, device=device)

    scores = [[
        [-2, 5, 5],
        [-7, 3, 8],
        [-7, -7, -8],
        ],[
        [6, 9, 8],  
        [3, 10, 7], 
        [2, 9, -9],
    ]]
    scores = torch.tensor(scores, dtype=dtype, device=device)

    print(f'tokens.shape {tokens.shape}')
    o = model(tokens, scores)
    print(f'tokens.shape {tokens.shape}')
    assert torch.allclose(o, tokens[:, :2, :])

if __name__ == '__main__':
    test_focal_prunning()