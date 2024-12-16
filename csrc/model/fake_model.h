#pragma once
#include <torch/torch.h>
#include "linear.h"

namespace mllm{

class FakeModelImpl : public torch::nn::Module{
public:
    int vocab_size_;
    torch::TensorOptions options_;
public:
    FakeModelImpl(int vocab_size, const torch::TensorOptions& options) : vocab_size_(vocab_size), options_(options){
    }
    torch::Tensor forward(torch::Tensor input_ids, torch::Tensor position_ids){
        int n_tokens = input_ids.size(0);
        return torch::randn({n_tokens, vocab_size_}, options_);
    }
};
TORCH_MODULE(FakeModel);

}