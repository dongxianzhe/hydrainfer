#pragma once

namespace mllm{

struct RequestOutput{
    std::string output_text;
    std::vector<int> output_token_ids;
    float arrival_time = 0.;
    float first_schedule_time = 0.;
    std::vector<int> tokens_times;
    float finished_time = 0.;
};

using OutputCallback = std::function<bool(RequestOutput output)>;
using BatchOutputCallback = std::function<bool(size_t index, RequestOutput output)>;
using OnOutput = std::function<bool(RequestOutput& output)>;

}