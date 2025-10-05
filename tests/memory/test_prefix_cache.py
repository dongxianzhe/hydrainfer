import pytest
from hydrainfer.memory.shared_cache import SharedCache, SharedCacheConfig, compute_hash


def test_basic_operation():
    cache = SharedCache(SharedCacheConfig(n_blocks=100))
    block_size = 4

    assert cache.match(compute_hash([1, 2, 3 ,4, 5, 6, 7, 8], block_size, -1)) == [-1, -1]
    assert cache.match(compute_hash([1, 2, 3 ,4], block_size, -1)) == [-1]
    assert cache.match(compute_hash([5, 6, 7, 8], block_size, -1)) == [-1]


    cache.pin([7, 11])
    cache.insert(compute_hash([1, 2, 3 ,4, 5, 6, 7, 8], block_size, -1), [7, 11])
    
    assert cache.match(compute_hash([1, 2, 3 ,4, 5, 6, 7, 8], block_size, -1)) == [7, 11]
    assert cache.match(compute_hash([3, 2, 3 ,4, 5, 6, 7, 8], block_size, -1)) == [-1, -1]
    assert cache.match(compute_hash([1, 2, 3 ,4, 6, 6, 7, 8], block_size, -1)) == [7, -1]
    assert cache.match(compute_hash([1, 2, 3 ,4], block_size, -1)) == [7]
    assert cache.match(compute_hash([5, 6, 7, 8], block_size, -1)) == [-1]

    assert len(cache.evict(2)) == 0
    assert len(cache.evict(1)) == 0

    cache.pin(cache.match(compute_hash([1, 2, 3, 4], block_size, -1)))

    cache.unpin(cache.match(compute_hash([1, 2, 3, 4, 5, 6, 7, 8], block_size, -1)))

    evicted_block_ids: list[int] = cache.evict(2)
    assert evicted_block_ids == [11]

    evicted_block_ids: list[int] = cache.evict(1)
    assert len(evicted_block_ids) == 0

    assert cache.match(compute_hash([1, 2, 3 ,4, 5, 6, 7, 8], block_size, -1)) == [7, -1]
    assert cache.match(compute_hash([3, 2, 3 ,4, 5, 6, 7, 8], block_size, -1)) == [-1, -1]
    assert cache.match(compute_hash([1, 2, 3 ,4, 6, 6, 7, 8], block_size, -1)) == [7, -1]
    assert cache.match(compute_hash([1, 2, 3 ,4], block_size, -1)) == [7]
    assert cache.match(compute_hash([5, 6, 7, 8], block_size, -1)) == [-1]

    cache.unpin(cache.match(compute_hash([1, 2, 3, 4], block_size, -1)))

    evicted_block_ids: list[int] = cache.evict(1)
    assert evicted_block_ids == [7]

    assert cache.match(compute_hash([1, 2, 3 ,4, 5, 6, 7, 8], block_size, -1)) == [-1, -1]
    assert cache.match(compute_hash([3, 2, 3 ,4, 5, 6, 7, 8], block_size, -1)) == [-1, -1]
    assert cache.match(compute_hash([1, 2, 3 ,4, 6, 6, 7, 8], block_size, -1)) == [-1, -1]
    assert cache.match(compute_hash([1, 2, 3 ,4], block_size, -1)) == [-1]
    assert cache.match(compute_hash([5, 6, 7, 8], block_size, -1)) == [-1]


if __name__ == '__main__':
    pytest.main([__file__, '-s'])
    