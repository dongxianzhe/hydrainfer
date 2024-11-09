#include<iostream>
#include<torch/torch.h>
#include"reduce_sum.cuh"

#define DISPATCH_PYTORCH_DTYPE_TO_CTYPE(pytorch_dtype, c_type, ...)                     \
  [&]{                                                                                  \
    switch (pytorch_dtype) {                                                            \
      case at::ScalarType::Float:{                                                      \
        using c_type = float;                                                           \
        __VA_ARGS__();                                                                  \
        break;                                                                          \
      }                                                                                 \
      case at::ScalarType::Half: {                                                      \
        using c_type = half;                                                            \
        __VA_ARGS__();                                                                  \
        break;                                                                          \
      }                                                                                 \
      default:                                                                          \
        throw std::runtime_error(" failed to dispatch data type ");                     \
    }                                                                                   \
  }()

void sum(at::Tensor& out, const at::Tensor& in){
    int n_elements = in.numel();
    TORCH_CHECK(in.scalar_type() == out.scalar_type());
    TORCH_CHECK(out.numel() == 1);
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE(in.scalar_type(), scalar_t, [&]{
        dim3 gridDim(1);
        dim3 blockDim(min(n_elements, 1024));
        sum_kernel<scalar_t><<<gridDim, blockDim>>>(
            static_cast<scalar_t*>(out.data_ptr()), 
            static_cast<scalar_t*>(in.data_ptr()), 
            n_elements
        );
    });
}