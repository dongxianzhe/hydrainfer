#include<torch/torch.h>
#include<glog/logging.h>
#include<cute/layout.hpp>
#include<cute/stride.hpp>
#include<cute/tensor.hpp>

template<int n_blocks>
__global__ void inc(half* a, int delta){
    using namespace cute;
    constexpr int warp_size = 32;
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    Tensor gA = make_tensor(make_gmem_ptr(a), make_shape(n_blocks, warp_size), make_stride(warp_size, 1));
#pragma unroll
    for(int i = 0;i < delta;i ++){
        gA(bid, tid) += 1;
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