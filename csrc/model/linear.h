#pragma once
namespace mllm{

class LinearImpl : public torch::nn::Module{
public:
    torch::Tensor weight_{nullptr};
    torch::Tensor bias_{nullptr};
    LinearImpl(int in_features, int out_features, bool bias, const torch::TensorOptions& options){
        weight_ = register_parameter("weight", torch::randn({out_features, in_features}, options), false);
        if(bias) bias_ = register_parameter("bias", torch::randn({out_features}, options), false);
    }
    torch::Tensor forward(torch::Tensor x){
        return torch::nn::functional::linear(x, weight_, bias_);
    }
};
TORCH_MODULE(Linear);

}