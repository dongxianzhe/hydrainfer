#pragma once
#include <torch/torch.h>

namespace mllm{

torch::Tensor linear_naive(torch::Tensor h, torch::Tensor w);
torch::Tensor linear_smem(torch::Tensor h, torch::Tensor w);
torch::Tensor linear_smem_v2(torch::Tensor h, torch::Tensor w);

using linear_func = torch::Tensor(*)(torch::Tensor, torch::Tensor);

}