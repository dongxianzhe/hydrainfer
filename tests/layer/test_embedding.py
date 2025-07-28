import torch
import pytest
from hydrainfer.layer.embedding import ParallelEmbedding
from hydrainfer.model_parallel.process_group import ProcessGroup
import torch.distributed as dist
import torch.multiprocessing as mp
import os

os.environ['MASTER_ADDR'] = '127.0.0.1'  
os.environ['MASTER_PORT'] = '29500'      

def worker(rank: int, world_size: int, vocab_size: int, embed_dim: int, dtype: torch.dtype, n_tokens: int):
    os.environ['RANK'] = str(rank)  
    print(f'rank {rank} world_size {world_size}')
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    print(f'rank {rank} world_size {world_size} init finished')
    process_group = ProcessGroup(rank, list(range(world_size)))
    parallelembedding = ParallelEmbedding(
        vocab_size = vocab_size, 
        embed_dim = embed_dim, 
        process_group = process_group
    )
    parallelembedding.to(dtype)
    parallelembedding.weight.fill_(value=rank)

    device = torch.device(f'cuda:{rank}')
    parallelembedding.to(device)
    x = torch.arange(n_tokens, dtype=torch.int, device=device)
    o = parallelembedding(x)
    assert o.shape == (n_tokens, embed_dim), f"{o.shape}"
    assert torch.allclose(o, torch.arange(world_size, dtype=dtype, device=device).view(1, world_size).repeat(n_tokens, 1).repeat_interleave(embed_dim // world_size, dim=1))

@pytest.mark.parametrize("vocab_size", [129280])
@pytest.mark.parametrize("embed_dim", [7168])
@pytest.mark.parametrize("dtype", [torch.half])
@pytest.mark.parametrize("world_size", [8])
@pytest.mark.parametrize("n_tokens", [512])
def test_parallel_embedding(
    vocab_size: int, 
    embed_dim: int, 
    dtype: torch.dtype, 
    world_size: int, 
    n_tokens: int, 
):
    os.environ['WORLD_SIZE'] = '8'          
    mp.spawn(worker, args=(world_size, vocab_size, embed_dim, dtype, n_tokens), nprocs=world_size, join=True)

if __name__ == '__main__':
    pytest.main([__file__, '-s'])