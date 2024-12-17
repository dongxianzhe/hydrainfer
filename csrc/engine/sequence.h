#pragma once
#include "stage.h"
#include <vector>

namespace mllm{

struct RequestOutput{
    std::vector<int> output_token_ids;
    float arrival_time = 0.;
    float first_schedule_time = 0.;
    std::vector<int> tokens_times;
    float finished_time = 0.;
};

struct Sequence{
    std::vector<Stage> stages;
    int curr_stage_id = 0;
    
    RequestOutput request_output;
    int max_tokens;
};

}