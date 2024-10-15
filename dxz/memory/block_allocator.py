class BlockAllocator:
    def __init__(self, total_blocks: int):
        self.total_blocks = total_blocks
        # free blocks' id list
        self.free_blocks: list[int] = [block_id for block_id in reversed(range(total_blocks))]

    def allocate(self, n_blocks: int) -> list[int]:
        assert n_blocks <= len(self.free_blocks), "not enough block to allocate"
        blocks: list[int] = []
        for _ in range(n_blocks):
            block_id = self.free_blocks.pop()
            blocks.append(block_id)
        return blocks
    
    def free(self, blocks: list[int]) -> bool:
        for block_id in blocks:
            assert block_id >= 0 and block_id < self.total_blocks, f"invalid block id {block_id}"
            self.free_blocks.append(block_id)