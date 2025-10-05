import numpy as np
import xxhash
from dataclasses import dataclass


@dataclass
class SharedBlock:
    ref_count: int
    


@dataclass
class SharedCacheConfig:
    n_blocks: int



class SharedCache:
    def __init__(self, config: SharedCacheConfig):
        # different block id may have same hash
        self.hash_to_block_id: dict[int, int] = {}
        self.block_id_to_hash: list[int] =  list(range(config.n_blocks))

        self.blocks: list[SharedBlock] = [SharedBlock(0) for i in range(config.n_blocks)]
        self.to_be_evicted = set()

    def match(self, hashes: list[int]) -> list[int]:
        return [self.hash_to_block_id.get(hash, -1) for hash in hashes]

    def pin(self, block_ids: list[int]):
        for block_id in block_ids:
            self.blocks[block_id].ref_count += 1
            assert self.blocks[block_id].ref_count > 0
            if block_id in self.to_be_evicted:
                self.to_be_evicted.remove(block_id)

    def unpin(self, block_ids: list[int]):
        for block_id in block_ids:
            self.blocks[block_id].ref_count -= 1
            assert self.blocks[block_id].ref_count >= 0
            if self.blocks[block_id].ref_count == 0:
                self.to_be_evicted.add(block_id)

    def insert(self, hashes: list[int], block_ids: list[int]):
        for hash, block_id in zip(hashes, block_ids):
            self.hash_to_block_id[hash] = block_id
            self.block_id_to_hash[block_id] = hash

    def evict(self, n_blocks: int) -> list[int]:
        # return at most n_blocks, less than n_blocks if there are not enough blocks to evict
        evicted_blocks: list[int] = []
        for i in range(min(n_blocks, len(self.to_be_evicted))):
            block_id = self.to_be_evicted.pop()
            hash = self.block_id_to_hash[block_id]
            evicted_blocks.append(block_id)
            if hash in self.hash_to_block_id:
                del self.hash_to_block_id[hash]
            self.block_id_to_hash[block_id] = -1
        return evicted_blocks


def compute_block_hash(token_ids: list[int], prefix: int = -1) -> int:
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()


def compute_hash(token_ids: list[int], block_size: int, prefix: int) -> list[int]:
    n_blocks = len(token_ids) // block_size
    hashes = []
    h = prefix
    for i in range(n_blocks):
        h = compute_block_hash(token_ids[i * block_size: i * block_size + block_size], prefix=h)
        hashes.append(h)
    return hashes

