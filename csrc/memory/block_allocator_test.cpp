#include "block_allocator.h"

#include <gtest/gtest.h>

namespace mllm{
TEST(BlockAllocatorTest, allocate){
    int total_blocks = 10;
    BlockAllocator allocator(total_blocks);
    // 9 8 7 6 5 4 3 2 1 0 -> 9 8 7 6 5 4 3 + 0 1 2
    std::vector<int> blocks = allocator.allocate(3);
    std::vector<int> ref_blocks = {0, 1, 2};

    for(int i = 0;i < blocks.size();i ++){
        EXPECT_EQ(blocks[i], ref_blocks[i]);
    }

    allocator.free({blocks[1]});
    allocator.free({blocks[2]});

    // 9 8 7 6 5 4 3 1 2 -> 9 8 7 6 5 4 + 2 1 3
    blocks = allocator.allocate(3);
    ref_blocks = {2, 1, 3};
    for(int i = 0;i < blocks.size();i ++){
        EXPECT_EQ(blocks[i], ref_blocks[i]);
    }
        
    blocks = allocator.allocate(0);
    allocator.free({});

    blocks = allocator.allocate(6);
    ref_blocks = {4, 5, 6, 7, 8, 9};
    for(int i = 0;i < blocks.size();i ++){
        EXPECT_EQ(blocks[i], ref_blocks[i]);
    }
}
}
