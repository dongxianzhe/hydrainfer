
from dataclasses import dataclass


class TrieTreeNode:
    def __init__(self):
        self.cnt: int = 0 # how many request is using this block
        self.block_id: int
        self.sons: dict[tuple, TrieTreeNode] = {} # token_ids -> TrieNode
        self.parent: TrieTreeNode


@dataclass
class PrefixCacheManagerConfig:
    n_blocks: int
    block_size: int


class PrefixCacheManager:
    def __init__(self, config: PrefixCacheManagerConfig) -> None:
        self.config = config
        self.nodes: list[TrieTreeNode] = [TrieTreeNode() for _ in range(self.config.n_blocks) + 10]
        self.root_node = self.nodes[0]
        self.free_node_ids: list[int] = [i for i in range(1, len(self.nodes))]

    def _round_down(self, n: int, mod: int):
        return n // mod * mod

    def _round_up(self, n: int, mod: int):
        return (n + mod - 1) // mod

    def match(self, token_ids: list[int]) -> list[int]:
        chunks = [tuple(token_ids[i: i + self.config.block_size]) for i in range(0, self._round_down(len(token_ids), self.config.block_size), self.config.block_size)]
        p = self.root_node
        cached_block_ids: list[int] = []
        for chunk in chunks:
            if chunk in p.sons:
                p = p.sons[chunk]
                cached_block_ids.append(p.block_id)
            else:
                break
        return cached_block_ids

    def insert(self, token_ids: list[int], block_ids: list[int]):
        assert self._round_up(len(token_ids), self.config.block_size) == len(block_ids)
        chunks = [tuple(token_ids[i: i + self.config.block_size]) for i in range(0, self._round_down(len(token_ids), self.config.block_size), self.config.block_size)]

        p = self.root_node
        for i, chunk in enumerate(chunks):
            if chunk in p.sons:
                p = p.sons[chunk]
                p.cnt += 1
            else:
                new_node = self.nodes[self.free_node_ids.pop()]
                new_node.block_id = block_ids[i]
                new_node.sons = {}
                p.sons[chunk] = new_node
                new_node.parent = p

                p = new_node
                p.cnt += 1