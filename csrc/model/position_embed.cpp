#include "position_embed.h"

namespace mllm{

torch::Tensor compute_default_inv_freq(int64_t rotary_dim, float theta) {
    CHECK(rotary_dim % 2 == 0) << "rotary_dim must be even";
    const auto slice = torch::arange(0, rotary_dim, 2, torch::kFloat32);
    return 1.0 / torch::pow(theta, slice / rotary_dim);
}

}