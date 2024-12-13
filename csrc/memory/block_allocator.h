#pragma once
#include<vector>
namespace mllm{

class BlockAllocator final{
public:
    BlockAllocator(int total_blocks);
    ~BlockAllocator();

    BlockAllocator(const BlockAllocator&) = delete;
    BlockAllocator(BlockAllocator&&) = delete;
    BlockAllocator& operator=(const BlockAllocator&) = delete;
    BlockAllocator& operator=(BlockAllocator&&) = delete;

    std::vector<int> allocate(int n_blocks);
    
    void free(const std::vector<int>& blocks);
private:
    int total_blocks_ = 0;
    std::vector<int> free_blocks_;
    int n_free_blocks_ = 0;
};

}