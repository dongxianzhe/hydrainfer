#include<iostream>
#include<torch/torch.h>

// __shfl_xor_syn(unsigined mask, T var, int lane_mask, int width)
// eg. __shfl_xor_sync(0xffffffff, val, 1, 32);
template<typename T>
__global__ void test_shfl_xor_sync_kernel(T* data, unsigned mask, int lane_mask, int width){
    T val = data[threadIdx.x];
    val = __shfl_xor_sync(mask, val, lane_mask, width);
    data[threadIdx.x] = val;
}

int main(int argc, char* argv[]){
    unsigned int mask = std::stoul(argv[1]);
    int lane_mask = std::atoi(argv[2]);
    int width = std::atoi(argv[3]);

    auto a = torch::arange(32, torch::kInt).to(torch::kCUDA);
    for(int i = 0;i < 32;i ++)printf("%d ", a[i].item<int>());puts("");

    test_shfl_xor_sync_kernel<<<1, 32>>>(static_cast<int*>(a.data_ptr()), mask, lane_mask, width);
    for(int i = 0;i < 32;i ++)printf("%d ", a[i].item<int>());puts("");
    return 0;
}