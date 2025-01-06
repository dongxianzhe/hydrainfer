#pragma once
#include <torch/torch.h>

namespace mllm{

torch::Tensor tile_linear(torch::Tensor h, torch::Tensor w);
using tile_linear_func = torch::Tensor(*)(torch::Tensor, torch::Tensor);

}