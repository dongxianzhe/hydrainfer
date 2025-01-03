#include<torch/torch.h>
#include <glog/logging.h>

template<int n_blocks>
__global__ void inc(half* a, int delta){
    int bid = blockIdx.x;
    int tid = threadIdx.x;
#pragma unroll
    for(int i = 0;i < delta;i ++){
        a[bid * blockDim.x + tid] ++;
    }
}

torch::Tensor inc(torch::Tensor a, int delta){
    constexpr int warp_size = 32;
    constexpr int num_blocks = 128;
    TORCH_CHECK(a.dtype() == torch::kHalf);
    TORCH_CHECK(a.is_cuda());
    TORCH_CHECK(a.is_contiguous());
    TORCH_CHECK(a.dim() == 2);
    TORCH_CHECK(a.size(1) == warp_size);
    TORCH_CHECK(a.size(0) == num_blocks);

    inc<num_blocks><<<num_blocks, warp_size>>>(
        static_cast<half*>(a.data_ptr()), 
        delta
    );
    
    return a;
}