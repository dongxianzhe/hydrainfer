#pragma once
#include "torch/torch.h"

namespace mllm{
    
class EmbeddingImpl : public torch::nn::Module{
public:
    torch::Tensor weight_{nullptr};
public:
    EmbeddingImpl(int num_embeddings, int embedding_dim, const torch::TensorOptions& options){
        weight_ = register_parameter("weight", torch::empty({num_embeddings, embedding_dim}, options), false);
    }
    torch::Tensor forward(torch::Tensor input){
        return torch::nn::functional::embedding(input, weight_);
    }
};
TORCH_MODULE(Embedding);

}