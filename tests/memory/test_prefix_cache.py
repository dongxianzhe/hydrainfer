import pytest
from hydrainfer.memory.shared_cache import SharedCache, SharedCacheConfig, compute_hash, compute_image_hash, compute_block_hash


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


from hydrainfer.utils.image_utils import make_random_image
import copy
def test_hash():
    image1 = make_random_image(height=336, width=336, n_channel=3)
    image2 = copy.deepcopy(image1)
    image3 = make_random_image(height=338, width=336, n_channel=3)
    image4 = make_random_image(height=338, width=336, n_channel=3)
    hash1 = compute_image_hash(image1)
    hash2 = compute_image_hash(image2)
    hash3 = compute_image_hash(image3)
    hash4 = compute_image_hash(image4)

    image_token_id = 100
    token_ids = [1, 2, 3, 4, 5, 6, 100, 100, 100, 10, 11, 12]
    token_ids1 = [hash1 if token_id == image_token_id else token_id for token_id in token_ids]
    token_ids2 = [hash2 if token_id == image_token_id else token_id for token_id in token_ids]
    token_ids3 = [hash3 if token_id == image_token_id else token_id for token_id in token_ids]
    token_ids4 = [hash4 if token_id == image_token_id else token_id for token_id in token_ids]
    token_ids5 = [13 if i == 0 else token_id for i, token_id in enumerate(token_ids)]

    hash0 = compute_block_hash(token_ids)
    hash1 = compute_block_hash(token_ids1)
    hash2 = compute_block_hash(token_ids2)
    hash3 = compute_block_hash(token_ids3)
    hash4 = compute_block_hash(token_ids4)
    hash5 = compute_block_hash(token_ids5)
    assert hash0 != hash1 # test w/o image 
    assert hash1 == hash2 # test same image and prompt
    assert hash1 != hash3 # test different image size
    assert hash3 != hash4 # test different image content
    assert hash1 != hash5 # test differnt prompt with same image


if __name__ == '__main__':
    pytest.main([__file__, '-s'])
    