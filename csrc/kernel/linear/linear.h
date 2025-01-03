#pragma once
#include <torch/torch.h>

namespace mllm{

torch::Tensor linear(torch::Tensor h, torch::Tensor w);

}