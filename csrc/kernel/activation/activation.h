#pragma once
#include <torch/torch.h>

namespace mllm::kernel{

torch::Tensor silu(torch::Tensor input);

}