#include <cuda_fp16.h>

template<typename T>
__inline__ __device__ T warp_reduce_sum(T val){
    val += __shfl_xor_sync(0xFFFFFFFF, val, 16, 32);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 8, 32);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 4, 32);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 2, 32);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 1, 32);
    return val;
}

template<typename T>
__inline__ __device__ T block_reduce_sum(T val){
    static __shared__ T shared[32]; // up to 32 warps in a block
    int lane_id = threadIdx.x & 0x1f;
    int warp_id = threadIdx.x >> 5;
    val = warp_reduce_sum<T>(val);
    if(lane_id == 0) shared[warp_id] = val;
    __syncthreads();
    if constexpr(std::is_same<T, half2>::value){
        val = threadIdx.x < (blockDim.x / 32.f) ? shared[lane_id] : make_half2(0.f, 0.f);
    }
    else{
        val = threadIdx.x < (blockDim.x / 32.f) ? shared[lane_id] : (T)(0.f);
    }
    val = warp_reduce_sum<T>(val);
    return val;
}
