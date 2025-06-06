import pytest
import torch
from torch import Tensor
from functools import partial
from hydrainfer.utils.benchmark_utils import benchmark
        
@pytest.mark.parametrize("n_tokens", [1, 10, 16, 128, 1024])
@pytest.mark.parametrize("n_experts", [4, 8, 16, 32, 64, 128, 256])
@pytest.mark.parametrize("topk", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float])
def test_topk_softmax(
    n_tokens: int, 
    n_experts: int, 
    topk: int, 
    dtype: torch.dtype, 
):
    device = torch.device('cuda:0')
    gating_logit = torch.randn(size=(n_tokens, n_experts), dtype=dtype, device=device)
    def topk_softmax_ref(gating_logit: Tensor) -> tuple[Tensor, Tensor]:
        weights_ref, indices_ref = torch.topk(torch.softmax(gating_logit, dim=-1), k=topk, dim=-1)
        indices_ref = indices_ref.to(torch.int)
        return weights_ref, indices_ref
    weights_ref, indices_ref = topk_softmax_ref(gating_logit)

    weights = torch.empty(size=(n_tokens, topk), dtype=dtype, device=device)
    indices = torch.empty(size=(n_tokens, topk), dtype=torch.int, device=device)
    from hydrainfer._C.kernel.moe import topk_softmax as topk_softmax_kernel
    def fused_topk_softmax(gating_logit: Tensor):
        topk_softmax_kernel(gating_logit, weights, indices)
    fused_topk_softmax(gating_logit)
    assert torch.allclose(weights, weights_ref)
    assert torch.equal(indices, indices_ref)

    # print(f'n_tokens {n_tokens} n_experts {n_experts} topk {topk} dtype {dtype}')
    # print(f'ref   {benchmark(partial(topk_softmax_ref, gating_logit=gating_logit))} s')
    # print(f'fused {benchmark(partial(fused_topk_softmax, gating_logit=gating_logit))} s')
    

@pytest.mark.parametrize("n_tokens", [1, 2, 16])
@pytest.mark.parametrize("dim", [16, 64])
@pytest.mark.parametrize("n_experts", [4, 8, 16])
@pytest.mark.parametrize("topk", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float, torch.half, torch.bfloat16])
@pytest.mark.parametrize("device",[torch.device('cuda:0')])
def test_permute_index(
    n_tokens: int, 
    dim: int, 
    n_experts: int, 
    topk: int, 
    dtype: torch.dtype, 
    device: torch.device, 
):
    def permute_index_ref(
        tokens: Tensor, # (n_tokens, dim)
        topk_indices: Tensor, # (n_tokens, topk)
    ) -> tuple[Tensor, Tensor]:
        n_tokens = tokens.shape[0]
        topk = topk_indices.shape[1]
        flatten_indices = topk_indices.view(-1) # (n_tokens * topk)
        # permuted_token_idx -> (n_token, topk) : (topk, 1)
        sorted_incices = flatten_indices.argsort(stable=True)
        sorted_incices = sorted_incices.to(torch.int)
        # permuted_token_idx -> token_id, [n_permuted_tokens]
        token_indices = sorted_incices.div(topk, rounding_mode="floor")
        # permuted_token_idx -> token_weight
        permuted_tokens = tokens[token_indices, :]
        return permuted_tokens, sorted_incices

    def unpermute_index_ref(
        permuted_tokens: Tensor, # [n_permuted_tokens, dim]
        sorted_incices: Tensor,  # [n_permuted_tokens]
        probs: Tensor,           # [n_token, topk]
        n_tokens: int, 
        topk: int, 
    ) -> Tensor:
        tokens = torch.zeros_like(permuted_tokens)
        # [n_permuted_tokens, dim] restore back to original order, sorted by (tokens)
        tokens[sorted_incices, :] = permuted_tokens
        # [n_permuted_tokens, dim] => [n_tokens, topk, dim]
        tokens = tokens.reshape(n_tokens, topk, -1)
        #   // [n_tokens, topk, dim] * [n_tokens, topk]
        tokens *= probs[:, :, None]
        # [n_tokens, dim], sum over topk
        return tokens.sum(dim=1)

    tokens = torch.randn(size=(n_tokens, dim), dtype=dtype, device=device)
    gating_logit = torch.randn(size=(n_tokens, n_experts), dtype=dtype, device=device)
    weights, indices = gating_logit.topk(topk, dim=-1)
    probs = weights.softmax(dim=-1)
    indices = indices.to(torch.int)

    permuted_tokens_ref, sorted_indices_ref = permute_index_ref(tokens, indices)
    from hydrainfer._C.kernel.moe import permute_with_index_map
    permuted_tokens, sorted_indices = permute_with_index_map(tokens, indices)

    assert torch.allclose(permuted_tokens_ref, permuted_tokens)
    # sorted_indices and sorted_indices is not equal because of kernel implementation

    from hydrainfer._C.kernel.moe import unpermute_with_index_map
    unpermute_out = unpermute_with_index_map(permuted_tokens, sorted_indices, probs)
    unpermute_out_ref = unpermute_index_ref(permuted_tokens_ref, sorted_indices_ref, probs, n_tokens, topk)
    assert torch.allclose(unpermute_out, unpermute_out_ref, atol=1e-2, rtol=1e-2)
    assert torch.allclose(tokens, unpermute_out, atol=1e-2, rtol=1e-2)

@pytest.mark.parametrize("n_tokens", [1, 2, 16])
@pytest.mark.parametrize("dim", [16, 64])
@pytest.mark.parametrize("n_experts", [4, 8, 16])
@pytest.mark.parametrize("topk", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float, torch.half, torch.bfloat16])
@pytest.mark.parametrize("device",[torch.device('cuda:0')])
def test_permute_mask(
    n_tokens: int, 
    dim: int, 
    n_experts: int, 
    topk: int, 
    dtype: torch.dtype, 
    device: torch.device, 
):
    def permute_mask_ref(
        tokens,      # [n_tokens, dim]
        routing_map  # [n_tokens, n_experts]
    ) -> tuple[Tensor, Tensor]:
        n_tokens, n_experts = routing_map.shape
        token_indices = torch.arange(n_tokens, dtype=torch.int, device=device)[None, :].expand(n_experts, n_tokens)
        sorted_incices = token_indices.masked_select(routing_map.t())
        permuted_tokens = tokens[sorted_incices, :]
        return permuted_tokens, sorted_incices

    def unpermute_mask_ref(
        permuted_tokens,  # [n_permuted_tokens, dim]
        permuted_probs,   # [n_permuted_tokens]
        sorted_incices,   # [n_permuted_tokens]
        n_tokens: int
    ) -> Tensor:
        n_permuted_tokens, dim = permuted_tokens.shape
        tokens = torch.zeros(size=(n_tokens, dim), dtype=dtype, device=device)
        index = sorted_incices[:, None].expand(-1, dim)
        # tokens[index[i][j]][j] = scr[i][j]
        tokens.scatter_add_(0, index.to(torch.int64), permuted_tokens * permuted_probs[:, None])
        return tokens

    tokens = torch.randn(size=(n_tokens, dim), dtype=dtype, device=device)
    gating_logit = torch.randn(size=(n_tokens, n_experts), dtype=dtype, device=device)
    weight, indices = gating_logit.topk(topk, dim=-1)
    probs = torch.zeros_like(gating_logit).scatter(dim=1, index=indices, value=1 / topk)
    routing_map = torch.zeros_like(gating_logit, dtype=torch.int).scatter(dim=1, index=indices, value=1).to(torch.bool)
    
    permuted_tokens_ref, row_id_map_ref = permute_mask_ref(tokens, routing_map)
    from hydrainfer._C.kernel.moe import permute_with_mask_map
    permuted_tokens, row_id_map = permute_with_mask_map(tokens, routing_map, topk)
    assert torch.allclose(permuted_tokens, permuted_tokens_ref)

    from hydrainfer._C.kernel.moe import unpermute_with_mask_map
    unpermute_out = unpermute_with_mask_map(permuted_tokens, row_id_map, probs)
    permuted_probs = probs.t().masked_select(mask=routing_map.t())
    unpermute_out_ref = unpermute_mask_ref(permuted_tokens_ref, permuted_probs, row_id_map_ref, n_tokens)
    assert torch.allclose(unpermute_out, unpermute_out_ref, rtol=1e-2, atol=1e-2)
    assert torch.allclose(tokens, unpermute_out, rtol=1e-2, atol=1e-2)


if __name__ == '__main__':
    pytest.main([__file__ + '::test_topk_softmax', '-x', '-s'])
    pytest.main([__file__ + '::test_permute_index', '-x', '-s'])
    pytest.main([__file__ + '::test_permute_mask', '-x', '-s'])