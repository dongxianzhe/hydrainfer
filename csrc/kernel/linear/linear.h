#pragma once
#include <torch/torch.h>

namespace mllm{

torch::Tensor linear_naive(torch::Tensor h, torch::Tensor w);
torch::Tensor linear(torch::Tensor h, torch::Tensor w);

using linear_func = torch::Tensor(*)(torch::Tensor, torch::Tensor);

}