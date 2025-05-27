import pytest
from hydrainfer.memory.block_allocator import BlockAllocator
import random

def test_allocate():
    total_blocks = 100
    allocator = BlockAllocator(total_blocks)
    # test random allocate
    free_blocks_ref: list[int] = [block_id for block_id in reversed(range(total_blocks))]
    for i in range(10):
        n = random.randint(1, 10)
        blocks = allocator.allocate(n)

        blocks_ref = []
        for _ in range(n):
            blocks_ref.append(free_blocks_ref.pop())
        blocks_ref.reverse()

        assert blocks == blocks_ref 

    # test allocate zero
    blocks = allocator.allocate(0)

def test_out_of_memory():
    total_blocks = 10
    allocator = BlockAllocator(total_blocks)
    print(allocator.allocate(11))
    assert allocator.allocate(11) == []
    assert allocator.allocate(2) == [1, 0]
    assert allocator.allocate(9) == []
    assert allocator.allocate(8) == [9, 8, 7, 6, 5, 4, 3, 2]
    assert allocator.allocate(1) == []

def test_free():
    total_blocks = 10
    allocator = BlockAllocator(total_blocks)
    assert allocator.allocate(3) == [2, 1, 0]
    allocator.free([1, 0])
    assert allocator.allocate(3) == [3, 1, 0]

if __name__ == '__main__':
    pytest.main([__file__])