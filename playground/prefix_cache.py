import time
from typing import Optional


# size_t PrefixCache::evict_helper(size_t n_blocks_to_evict) {
#   size_t total_evicted = 0;
#   // evict nodes at the end to avoid invaliding iterator
#   std::vector<Node*> nodes_to_evict;
#   int64_t pre_access_time = 0;
#   for (Node* node = lru_front_.next;
#        total_evicted < n_blocks_to_evict && node != &lru_back_;
#        node = node->next) {
#     CHECK(pre_access_time <= node->last_access_time)
#         << "The last access time should be in ascending order";
#     pre_access_time = node->last_access_time;

#     // skip non-leaf nodes
#     if (!node->children.empty()) {
#       continue;
#     }

#     // find first non-shared block to evict
#     const auto& blocks = node->blocks;
#     const size_t n_blocks = blocks.size();
#     size_t non_shared_start = 0;
#     for (; non_shared_start < n_blocks; ++non_shared_start) {
#       if (!blocks[non_shared_start].is_shared()) {
#         break;
#       }
#     }

#     // try to only evict minimal number of blocks
#     const size_t n_to_evict = std::min(n_blocks_to_evict - total_evicted,
#                                        n_blocks - non_shared_start);
#     total_evicted += n_to_evict;
#     if (n_to_evict == n_blocks) {
#       // mark the node as to be evicted
#       nodes_to_evict.push_back(node);
#     } else if (n_to_evict > 0) {
#       // partially evict non-shared blocks
#       const size_t n_blocks_left = n_blocks - n_to_evict;
#       DCHECK(n_blocks_left >= non_shared_start);
#       node->token_ids.resize(n_blocks_left * block_size_);
#       node->blocks.resize(n_blocks_left);
#     }
#   }

#   // release leaf nodes and update leaf_nodes_ set
#   for (Node* node : nodes_to_evict) {
#     release_node(node);
#   }

#   // update the number of blocks
#   num_blocks_ -= total_evicted;
#   return total_evicted;
# }

# void PrefixCache::release_node(Node* node) {
#   DCHECK(node != &root_);
#   DCHECK(node->children.empty()) << "should only release leaf node";
#   // remove the node from the parent's children
#   auto* parent = node->parent;
#   DCHECK(parent->children.count(node) > 0);
#   parent->children.erase(node);

#   // delete the node
#   remove_node_from_lru(node);
#   delete node;
#   --num_nodes_;
# }

class Node:
    def __init__(self) -> None:
        self.token_ids: list[int] = []
        self.block_ids: list[int] = []
        # radix tree pointer
        self.children: list[Node] = []
        self.parent: Optional[Node] = None
        # lru list pointer
        self.last_access_time: float= 0.
        self.prev: Optional[Node] = None
        self.next: Optional[Node] = None

class PrefixCache:
    def __init__(self, n_blocks: int, block_size: int) -> None:
        assert n_blocks > 0, f"n_blocks should greater than 0, got {n_blocks}"
        self.n_blocks = n_blocks
        self.block_size = block_size
        self.nodes_pool: list[Node] = [Node() for _ in range(n_blocks + 10)]
        self.node_allocator = -1
        # None <- lru_front <-> ... <-> lru_back -> None
        # <- prev ------ next ->
        # evict from lru_front
        self.lru_front = self._new_node(token_ids=[], block_ids=[])
        self.lru_back = self._new_node(token_ids=[], block_ids=[])
        self.lru_front.next = self.lru_back
        self.lru_back.prev = self.lru_front
        self.root = self._new_node(token_ids=[], block_ids=[])

    def _common_prefix_length(self, token_ids1: list[int], token_ids2: list[int]) -> int:
        i = 0
        while i < len(token_ids1) and i < len(token_ids2) and token_ids1[i] == token_ids2[i]:
            i += 1
        return i

    def _round_down(self, n: int, mod: int) -> int:
        return n // mod * mod

    def _new_node(self, token_ids: list[int], block_ids: list[int]) -> Node:
        self.node_allocator += 1
        node = self.nodes_pool[self.node_allocator]
        node.next = node.prev = None
        node.last_access_time = 0.
        node.token_ids = token_ids
        node.block_ids = block_ids
        node.children = []
        node.parent = None
        return 

    def _add_node_to_lru_back(self, node: Node):
        node.prev = self.lru_back.prev
        node.next = self.lru_back
        self.lru_back.prev.next = node
        self.lru_back.prev = node

    def _remove_node_from_lru(self, node: Node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def _move_node_to_lru_back(self, node: Node):
        self._remove_node_from_lru(node)
        self._add_node_to_lru_back(node)

    def _split_node(self, node: Node, common_prefix_length: int):
        assert common_prefix_length > 0 and common_prefix_length % self.block_size == 0
        common_prefix_blocks = common_prefix_length // self.block_size
        assert len(node.token_ids) > common_prefix_length and len(node.block_ids) > common_prefix_blocks
        child = self._new_node()
        self._add_node_to_lru_back(child)
        child.token_ids = node[common_prefix_length:]
        child.block_ids = node[common_prefix_blocks:]
        child.last_access_time = node.last_access_time
        child.parent = node
        child.children = node.children
        for grand_child in child.children:
            grand_child.parent = child
        node.token_ids = node.token_ids[:common_prefix_length]
        node.block_ids = node.token_ids[:common_prefix_blocks]
        node.children.append(child)

    def match(self, token_ids: list[int]) -> list[int]:
        # match the token ids with the prefix tree
        # return matched blocks
        now = time.monotonic()
        block_ids: list[int] = []
        left = 0
        right = self._round_down(len(token_ids), self.block_size)
        next_node: Optional[Node] = self.root
        while next_node is not None and left < right:
            curr: Node = next_node
            next_node = None
            for child in curr.children:
                i = 0
                while i < len(child.token_ids) and left + i < right and child.token_ids[i] == token_ids[left + i]:
                    i += 1
                prefix_length = self._round_down(i, self.block_size)
                if prefix_length == 0:
                    continue
                child.last_access_time = now
                self._move_node_to_lru_back(child)
                block_ids.extend(child.block_ids[: prefix_length // self.block_size])
                left += prefix_length
                if prefix_length == len(child.token_ids):
                    next_node = child
                else:
                    self._split_node(child, prefix_length)
                break
        return block_ids

    def insert(self, token_ids: list[int], block_ids: list[int]) -> int:
        # insert the token ids and blocks into the prefix tree
        # return the length of new inserted tokens
        now = time.monotonic()
        n_blocks = len(token_ids) // self.block_size
        n_tokens = n_blocks * self.block_size
        left = 0
        right = n_tokens
        new_insert_tokens: int = 0
        next_node: Optional[Node] = self.root
        while next_node is not None and left < right:
            curr: Node = next_node
            next_node = None
            for child in curr.children:
                i = 0
                while i < len(child.token_ids) and left + i < right and child.token_ids[i] == token_ids[left + i]:
                    i += 1
                prefix_length = self._round_down(i, self.block_size)
                if prefix_length == 0:
                    continue
                next_node = child
                child.last_access_time = now
                self._move_node_to_lru_back(child)
                left += prefix_length
                if prefix_length < len(child.token_ids):
                    self._split_node(child, prefix_length)
                break
            if next_node is None:
                node = self._new_node(
                    token_ids=token_ids[left: right], 
                    block_ids=block_ids[left // self.block_size: right // self.block_size]
                )
                new_insert_tokens += right - left
                node.last_access_time = now
                node.parent = curr
                curr.children.append(node)
                self._add_node_to_lru_back(node)
        return new_insert_tokens 

    def evict(self, n_blocks_to_evict: int) -> list[int]:
        # evict blocks hold by the prefix cache
        # return the actual number of evicted blocks
        total_evicted: int = 0
        while total_evicted < n_blocks_to_evict:
            evicted = self._evict_leaf_node(n_blocks_to_evict - total_evicted)
            if evicted == 0:
                break
            total_evicted += evicted
        return total_evicted

    def _evict_leaf_node(self, n_blocks_to_evict: int):
        total_evicted: int = 0
        pre_access_time: float = 0.
        nodes_total_evicted: list[Node] = []
        node: Node = self.lru_front.next
        while total_evicted < n_blocks_to_evict and node is not self.lru_back:
            assert pre_access_time <= node.last_access_time, "The last access time should be in ascending order"
            if len(node.children) > 0:
                continue
            pre_access_time = node.last_access_time
            node = node.next

# size_t PrefixCache::evict_helper(size_t n_blocks_to_evict) {
#   for (Node* node = lru_front_.next;
#        total_evicted < n_blocks_to_evict && node != &lru_back_;
#        node = node->next) {
#     // find first non-shared block to evict
#     const auto& blocks = node->blocks;
#     const size_t n_blocks = blocks.size();
#     size_t non_shared_start = 0;
#     for (; non_shared_start < n_blocks; ++non_shared_start) {
#       if (!blocks[non_shared_start].is_shared()) {
#         break;
#       }
#     }

#     // try to only evict minimal number of blocks
#     const size_t n_to_evict = std::min(n_blocks_to_evict - total_evicted,
#                                        n_blocks - non_shared_start);
#     total_evicted += n_to_evict;
#     if (n_to_evict == n_blocks) {
#       // mark the node as to be evicted
#       nodes_to_evict.push_back(node);
#     } else if (n_to_evict > 0) {
#       // partially evict non-shared blocks
#       const size_t n_blocks_left = n_blocks - n_to_evict;
#       DCHECK(n_blocks_left >= non_shared_start);
#       node->token_ids.resize(n_blocks_left * block_size_);
#       node->blocks.resize(n_blocks_left);
#     }
#   }

#   // release leaf nodes and update leaf_nodes_ set
#   for (Node* node : nodes_to_evict) {
#     release_node(node);
#   }

#   // update the number of blocks
#   num_blocks_ -= total_evicted;
#   return total_evicted;
# }


if __name__ == '__main__':
