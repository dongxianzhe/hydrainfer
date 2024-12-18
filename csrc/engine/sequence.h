#pragma once
#include "stage.h"
#include <vector>
#include "request.h"

namespace mllm{

struct SamplingParams{
    int max_tokens = -1;
};


struct Sequence{
    std::vector<Stage> stages;
    int curr_stage_id = 0;
    std::vector<int> block_tables;
    int n_kv_cache_tokens = 0;
    
    RequestOutput request_output;
    int max_tokens;
    OnOutput on_output;
};

}