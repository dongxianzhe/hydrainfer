class BlockAllocator:
    def __init__(self, total_blocks: int):
        self.total_blocks = total_blocks
        # free blocks' id list
        self.free_blocks: list[int] = [block_id for block_id in reversed(range(total_blocks))]

    def allocate(self, n_blocks: int) -> list[int]:
        if n_blocks == 0:
            return []
        if n_blocks > len(self.free_blocks):
            raise Exception(f'not enough blocks, need {n_blocks}')
            # not enough block to allocate
            return []

        blocks, self.free_blocks = self.free_blocks[-n_blocks:], self.free_blocks[:-n_blocks]
        return blocks
    
    def free(self, blocks: list[int]):
        self.free_blocks += blocks
        assert len(self.free_blocks) <= self.total_blocks