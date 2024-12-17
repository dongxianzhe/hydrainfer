#pragma once
#include "attention_params.h"

namespace mllm{

class ModelParameters{
public:
    std::vector<AttentionParameters> attention_params;
};

}