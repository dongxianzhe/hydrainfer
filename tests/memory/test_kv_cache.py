from dxz.memory.kv_cache import KVCache
import torch

device = torch.device('cuda:0')

if __name__ == '__main__':
    num_blocks = 3
    block_size = 4
    num_kv_heads = 2
    head_size = 2 

    key_cache = torch.zeros(size=(num_blocks, block_size, num_kv_heads, head_size), device=device)
    value_cache = torch.zeros(size=(num_blocks, block_size, num_kv_heads, head_size), device=device)
    kv_cache = KVCache(key_cache, value_cache)
    
    num_tokens = 3
    slot_ids = torch.tensor([2, 3, 4], dtype=torch.int, device=device)
    keys = torch.ones(size=(num_tokens, num_kv_heads, head_size), device=device)
    values = 2 * torch.ones(size=(num_tokens, num_kv_heads, head_size), device=device)
    kv_cache.set_kv_cache(slot_ids=slot_ids, keys=keys, values=values)

    print(kv_cache.get_kv_cache()[0])
    print(kv_cache.get_kv_cache()[1])