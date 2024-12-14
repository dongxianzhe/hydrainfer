#pragma once
#include <torch/torch.h>

namespace mllm::kernel{

void rms_norm(torch::Tensor& out,
              torch::Tensor input,
              torch::Tensor weight,
              float epsilon);

}