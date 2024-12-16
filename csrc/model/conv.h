#pragma once
#include <torch/torch.h>
#include <glog/logging.h>

namespace mllm{

class Conv2dImpl : public torch::nn::Module{
public:
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    bool bias_;
    
    torch::Tensor weight_{nullptr};
    // torch::Tensor bias_{nullptr};
public:
    Conv2dImpl(int in_channels, int out_channels, int kernel_size, int stride, bool bias, const torch::TensorOptions& options) : in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size), stride_(stride), bias_(bias){
        weight_ = register_parameter("weight", torch::randn({out_channels, in_channels, kernel_size, kernel_size}, options), false);
        CHECK(bias == false) << "not implemented bias for conv2d";
        // bias_ = register_parameter("bias", torch::zeros({out_channels}, options), false);
    }
    torch::Tensor forward(torch::Tensor x){
        torch::nn::functional::Conv2dFuncOptions options;
        return torch::nn::functional::conv2d(x, weight_, options.stride(stride_).padding(0));
    }
};

TORCH_MODULE(Conv2d);

}