#include<iostream>
#include<torch/torch.h>
#include"reduce_sum.cuh"
#include"dispatch.h"

void sum(at::Tensor& out, const at::Tensor& in){
    int n_elements = in.numel();
    TORCH_CHECK(in.scalar_type() == out.scalar_type());
    TORCH_CHECK(out.numel() == 1);
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE(in.scalar_type(), scalar_t, [&]{
        DISPATCH_BOOL(n_elements % 2 == 0, is_even_n, [&]{
            dim3 gridDim(1);
            dim3 blockDim;
            if constexpr(std::is_same<scalar_t, half>::value && is_even_n){
                blockDim = min(n_elements / 2, 1024);
            }else{
                blockDim = min(n_elements, 1024);
            }
            sum_kernel<scalar_t, is_even_n><<<gridDim, blockDim>>>(
                static_cast<scalar_t*>(out.data_ptr()), 
                static_cast<scalar_t*>(in.data_ptr()), 
                n_elements
            );
        });
    });
}