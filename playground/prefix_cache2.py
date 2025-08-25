from typing import Optional

class Node:
    def __init__(self):
        self.token_ids: list[int] = []
        self.block_id: int = -1
        self.reference_cnt: int = 0
        # trie tree pointer
        self.last_access_time: float = 0.
        self.children: list[Node] = []
        self.parent: Optional[Node] = None
        # lru list
        self.prev: Optional[Node] = None
        self.next: Optional[Node] = None

class NodePool:
    def __init__(self, n_nodes: int) -> None:
        self.n_nodes = n_nodes
        self.nodes = [Node() for _ in range(n_nodes)]

    def new_node(self) -> Node:
        node = self.nodes.pop()
        node.token_ids = []
        node.block_id = -1
        node.reference_cnt = 0
        node.last_access_time = 0.
        node.children = []
        node.parent = None
        node.prev = None
        node.next = None
        return node

    def free_node(self, node: Node):
        self.nodes.append(node)
        assert len(self.nodes) <= self.n_nodes

class LRUList:
    """
    None <- lru_front <-> ... <-> lru_back -> None
    <- prev
    next ->
    """
    def __init__(self, node_pool: NodePool) -> None:
        self.node_pool = node_pool
        self.lru_back = self.node_pool.new_node()
        self.lru_front = self.node_pool.new_node()
        self.lru_front.prev = None
        self.lru_front.next = self.lru_back
        self.lru_back.prev = self.lru_front
        self.lru_back.next = None

    def update(self, node: Node):
        if node.prev is not None:
            node.prev.next = node.next
        if node.next is not None:
            node.next.prev = node.prev
        node.prev = self.lru_back.prev
        node.next = self.lru_back
        self.lru_back.prev.next = node
        self.lru_back.prev = node

    def evict(self, node: Node):
        if node.prev is not None:
            node.prev.next = node.next
        if node.next is not None:
            node.next.prev = node.prev
        node.prev = None
        node.next = None

class PrefixCache:
    def __init__(self, n_blocks: int, block_size: int) -> None:
        assert n_blocks > 0
        self.n_blocks = n_blocks
        self.block_size = block_size
        self.node_pool = NodePool(n_nodes=n_blocks + 10)
        self.lru_list = LRUList(node_pool=self.node_pool)
        self.block_id2node: dict[int, Node] = {}
        self.root = self.node_pool.new_node()
    
    def insert(self, token_ids: list[int], block_ids: list[int]):
        n_blocks: int = len(token_ids) // self.block_size
        curr = self.root
        for i in range(n_blocks):
            block_token_ids: list[int] = token_ids[i * self.block_size: (i + 1) * self.block_size]
            block_id: int = block_ids[i]
            next_node: Optional[Node] = None
            for child in curr.children:
                if child.token_ids == block_token_ids and child.block_id == block_id:
                    next_node = child
                    break
            if next_node is None:
                next_node = self.node_pool.new_node()
                next_node.token_ids = block_token_ids
                next_node.block_id = block_id
                next_node.reference_cnt = 0
                curr.children.append(next_node)
                next_node.parent = curr
                self.block_id2node[block_id] = next_node
            curr = next_node
        
        while curr is not self.root:
            curr.reference_cnt += 1
            self.lru_list.update(curr)
            curr = curr.parent

    def query(self, token_ids: list[int]) -> list[int]: 
        n_blocks = len(token_ids) // self.block_size
        curr: Node = self.root
        block_ids: list[int] = []
        for i in range(n_blocks):
            block_token_ids: list[int] = token_ids[i * self.block_size: (i + 1) * self.block_size]
            next_node: Optional[Node] = None
            for child in curr.children:
                if child.token_ids == block_token_ids:
                    next_node = child
                    break
            if next_node:
                block_ids.append(next_node.block_id)
                curr = next_node
        while curr is not self.root:
            self.lru_list.update(curr)
            curr = curr.parent
        return block_ids

    def free(self, block_ids: list[int]):
        for block_id in block_ids:
            node = self.block_id2node[block_id]
            node.reference_cnt -= 1
            assert node.reference_cnt >= 0

    def _evict_leaf_node(self, n_blocks: int) -> list[int]:
        curr = self.lru_list.lru_front.next
        block_ids: list[int] = []
        while curr is not self.lru_list.lru_back and len(block_ids) < n_blocks:
            if len(curr.children) > 0:
                # assert curr.next is not None
                curr = curr.next
            next_node = curr.next
            if curr.reference_cnt == 0:
                block_ids.append(curr.block_id)
                self.lru_list.evict(curr)
                for i, child in enumerate(curr.parent.children):
                    if id(child) == id(curr):
                        del curr.parent.children[i]
                self.node_pool.free_node(curr)
                del self.block_id2node[curr.block_id]
            # assert curr.next is not None
            curr = next_node
        return block_ids

    def evict(self, n_blocks: int) -> list[int]:
        block_ids: list[int] = []
        while len(block_ids) < n_blocks:
            leaf_block_ids: list[int] = self._evict_leaf_node(n_blocks - len(block_ids))
            if len(leaf_block_ids) == 0:
                break
            block_ids.extend(leaf_block_ids)
        return block_ids

    def num_nodes(self) -> int:
        curr = self.lru_list.lru_front.next
        cnt: int = 0
        while curr is not self.lru_list.lru_back:
            cnt += 1
            print(f"({curr.token_ids} {curr.block_id})")
            curr = curr.next
        print('------------------------------')
        return cnt

def test_empty_cache():
    prefix_cache = PrefixCache(n_blocks=100, block_size=2)
    token_ids: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    block_ids: list[int] = prefix_cache.query(token_ids)
    assert block_ids == [], f"block_ids {block_ids}"

def test_insert_query():
    cache = PrefixCache(n_blocks=100, block_size=2)
    token_ids = [1, 2, 3, 4, 5, 6]
    block_ids = [0, 1, 2]
    cache.insert(token_ids, block_ids)
    n_nodes = cache.num_nodes()
    assert n_nodes == 3 , f"cache nodes {n_nodes}"

    token_ids = [1, 2, 3, 4, 50, 60, 70, 80, 90, 100, 110]
    block_ids = [0, 1, 20, 30, 40, 50]
    cache.insert(token_ids, block_ids)
    n_nodes = cache.num_nodes()
    # expected two sequences split at [1, 2, 3, 4]
    #    tokens: [1, 2, 3, 4] -> [5, 6]*
    #                         -> [50, 60, 70, 80, 90, 100]*
    #    blocks: [0, 1] -> [2]*
    #                   -> [20, 30, 40]*
    assert n_nodes == 6 , f"cache nodes {n_nodes}"

    token_ids = [1, 2, 5, 6, 7, 8, 9, 10, 11]
    block_ids = [0, 5, 15, 25, 35]
    len = cache.insert(token_ids, block_ids);
    n_nodes = cache.num_nodes()
    assert n_nodes == 9 , f"cache nodes {n_nodes}"
    # expect 3 sequences split at [1, 2]
    #   tokens: [1, 2] -> [5, 6, 7, 8, 9, 10]*
    #                  -> [3, 4] -> [5, 6]*
    #                            -> [50, 60, 70, 80, 90, 100]*
    #   blocks: [0] -> [5, 15, 25]*
    #               -> [1] -> [2]*
    #                      -> [20, 30, 40]*

    # no match
    token_ids = [3, 4, 5, 6, 7, 8, 9, 10]
    block_ids = cache.query(token_ids)
    assert block_ids == [], f"block_ids {block_ids}"

    # match first sequence partially
    token_ids = [1, 2, 5, 6, 8]
    block_ids = cache.query(token_ids)
    assert block_ids == [0, 5], f"block_ids {block_ids}"

    # match second sequence fully
    token_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    block_ids = cache.query(token_ids)
    assert block_ids == [0, 1, 2], f"block_ids {block_ids}"

    # match third sequence partially
    token_ids = [1, 2, 3, 4, 50, 60, 70, 80, 90]
    block_ids = cache.query(token_ids)
    assert block_ids == [0, 1, 20, 30], f"block_ids {block_ids}"

def test_concurrent_request_free():
    cache = PrefixCache(n_blocks=100, block_size=2)
    cache.insert([30, 40, 50, 60], [1, 2])
    cache.insert([30, 40, 50, 60, 70, 80], [3, 4, 5])
    n_nodes = cache.num_nodes()
    assert n_nodes == 5, f"n_nodes {n_nodes}"
    block_ids = cache.query([30, 40, 50, 60, 70, 80, 90, 100])
    assert block_ids == [1, 2], f"block_ids {block_ids}"
    cache.free([1, 2])
    block_ids = cache.query([30, 40, 50, 60, 70, 80, 90, 100])
    assert block_ids == [1, 2], f"block_ids {block_ids}"
    block_ids = cache.evict(3)
    assert block_ids == [2, 1], f"block_ids {block_ids}"
    block_ids = cache.query([30, 40, 50, 60, 70, 80, 90, 100])
    assert block_ids == [3, 4, 5], f"block_ids {block_ids}"

if __name__ == "__main__":
    test_empty_cache()
    test_insert_query()
    test_concurrent_request_free()

    max_seq_len = 2048
    block_size = 16
    n_blocks = 8192
    vocab_size = 16
    num_operations = 10000


    cache = PrefixCache(n_blocks=n_blocks, block_size=block_size)
    import random
    from dataclasses import dataclass
    for i in range(num_operations):
        op = random.randint(0, 3)
        # insert
        if op == 0:
        # query
        elif op == 1:
        # free
        elif op == 2:
        # evict
        else:


#   BlockAllocator allocator(total_blocks, block_size);
#   PrefixCache cache(block_size);

#   absl::BitGen gen;
#   // construct sequences and insert into prefix cache
#   std::vector<SequenceData> seqs_data;
#   for (int i = 0; i < num_seqs; i++) {
#     {
#       // generate random sequence
#       // which seq to get common prefix, seq_idx == -1, no common prefix
#       int32_t seq_idx = i == 0 ? -1 : absl::Uniform<int32_t>(gen, 0, num_seqs);

#       // generate token ids
#       std::vector<int32_t> token_ids;
#       std::vector<Block> blocks;
#       int32_t prefix_len = 0;
#       // get common prefix
#       if (seq_idx < seqs_data.size()) {
#         // common prefix len
#         prefix_len =
#             absl::Uniform<int32_t>(gen, 0, seqs_data[seq_idx].token_ids.size());
#         token_ids = sub_vector(seqs_data[seq_idx].token_ids, prefix_len);
#       }
#       // total seq len
#       int32_t seq_len = absl::Uniform<int32_t>(gen, prefix_len, max_seq_len);
#       // generate rest of the sequence
#       for (size_t j = token_ids.size(); j < seq_len; j++) {
#         token_ids.push_back(absl::Uniform<int32_t>(gen, 0, vocab_size));
#       }

#       // get shared blocks from prefix cache
#       blocks = cache.match(token_ids);

#       // allocate blocks for rest of the sequence
#       size_t num_blocks = (seq_len + block_size - 1) / block_size;
#       for (size_t j = blocks.size(); j < num_blocks; j++) {
#         blocks.push_back(allocator.allocate());
#       }

#       // insert the sequence and blocks into prefix cache
#       cache.insert(token_ids, blocks);

#       size_t cached_len = seq_len / block_size;
#       blocks.resize(cached_len);

#       // query back and check
#       std::vector<Block> matched_blocks = cache.match(token_ids);
#       EXPECT_EQ(matched_blocks, blocks);

#       // save the sequence and blocks
#       seqs_data.push_back({token_ids, blocks});
#     }

#     // all blocks either in cache or allocator
#     ASSERT_EQ(cache.num_blocks() + allocator.num_free_blocks(), total_blocks);
#   }

#   // randomly query the prefix cache and compare the result with the saved
#   for (int i = 0; i < 1000; i++) {
#     const int32_t seq_idx = absl::Uniform<int32_t>(gen, 0, num_seqs);
#     const int32_t seq_len =
#         absl::Uniform<int32_t>(gen, 0, seqs_data[seq_idx].token_ids.size());

#     // randomly generate partial sequence
#     std::vector<int32_t> token_ids =
#         sub_vector(seqs_data[seq_idx].token_ids, seq_len);
#     std::vector<Block> desired_blocks =
#         sub_vector(seqs_data[seq_idx].blocks, seq_len / block_size);

#     // match the sequence and compare the result
#     std::vector<Block> blocks = cache.match(token_ids);
#     EXPECT_EQ(blocks, desired_blocks);
#   }

#   // can't evict any blocks since all blocks hold by seqs_data
#   ASSERT_EQ(cache.evict(100), 0);
#   // release hold blocks
#   seqs_data.clear();

#   // randomly evict all blocks
#   int32_t blocks_left = cache.num_blocks();
#   while (blocks_left > 0) {
#     // randomly generate number of blocks to evict this round: [1, blocks_left]
#     int32_t to_evict = absl::Uniform<int32_t>(gen, 1, blocks_left + 1);
#     int32_t evicted = cache.evict(to_evict);
#     // evicted should be non-zero, otherwise, it's a deadloop
#     ASSERT_GT(evicted, 0);
#     // should evicted exactly same number of blocks since no thers hold blocks
#     EXPECT_EQ(to_evict, evicted);
#     blocks_left -= evicted;
#   }

#   // all blocks are evicted and return to allocator
#   EXPECT_EQ(cache.num_blocks(), 0);
#   EXPECT_EQ(allocator.num_free_blocks(), total_blocks);
# }

# INSTANTIATE_TEST_SUITE_P(
#     Random,
#     PrefixCacheRandomTest,
#     ::testing::Combine(::testing::Values(1, 4, 8, 32, 128, 256),  // block_size
#                        ::testing::Values(1000),                   // max_seq_len
#                        ::testing::Values(1000)                    // num_seqs
#                        ));

# }  // namespace llm