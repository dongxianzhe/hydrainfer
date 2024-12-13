#include "block_allocator.h"

#include <glog/logging.h>

#include<vector>
namespace mllm{

BlockAllocator::BlockAllocator(int total_blocks) : total_blocks_(total_blocks), n_free_blocks_(total_blocks){
    CHECK_GT(total_blocks, 0) << "no blocks to allocate";
    free_blocks_.reserve(total_blocks);
    for(int i = total_blocks - 1;i >= 0;i --){
        free_blocks_.push_back(i);
    }
}

BlockAllocator::~BlockAllocator() {
    CHECK(free_blocks_.size() == total_blocks_) << "not all blocks are freed";
};

std::vector<int> BlockAllocator::allocate(int n_blocks){
    std::vector<int> blocks(n_blocks);
    for(int i = 0;i < n_blocks;i ++){
        CHECK_GT(n_free_blocks_, 0) << "not enough blocks";
        blocks[i] = free_blocks_[--n_free_blocks_];
    }
    return blocks;
}
    
void BlockAllocator::free(const std::vector<int>& blocks){
    for(int i = 0;i < blocks.size();i ++){
        CHECK_LT(n_free_blocks_, total_blocks_) << "free blocks overflow";
        free_blocks_[n_free_blocks_ ++] = blocks[i];
    }
}

}
