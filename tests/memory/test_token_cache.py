import pytest
import torch
from hydrainfer.memory import TokenCacheBlockManager, TokenCacheBlockManagerConfig, TokenCacheBlockManagerContext


def test_token_cache_manager():
    device_str = 'cuda:0'
    device = torch.device(device_str)
    max_memory = torch.cuda.get_device_properties(device).total_memory
    print(f'max_memory {max_memory}B')
    config = TokenCacheBlockManagerConfig(
        n_layers = 32, 
        n_tokens = 2, 
        block_size = 16, 
        n_heads = 32, 
        head_size = 128, 
        dtype = "fp16", 
        device = device_str, 
    )
    memory_allocated_ratio = 0.9
    context = TokenCacheBlockManagerContext(
        n_blocks = TokenCacheBlockManager.compute_n_blocks(config, int(max_memory * memory_allocated_ratio)), 
        rank=0
    )
    manager = TokenCacheBlockManager(config, context)

    def check_memory_allocated(expected_ratio: float):
        print(f'actual memory_allocated_ratio {torch.cuda.memory_allocated() / max_memory}')
        assert abs(torch.cuda.memory_allocated() / max_memory - expected_ratio) < 0.01 < 0.1
    check_memory_allocated(memory_allocated_ratio)

    new_memory_allocated_ratio = 0.3
    context.n_blocks = TokenCacheBlockManager.compute_n_blocks(config, int(max_memory * new_memory_allocated_ratio))
    manager.update(context)
    check_memory_allocated(new_memory_allocated_ratio)


if __name__ == '__main__':
    pytest.main([__file__, '-s'])