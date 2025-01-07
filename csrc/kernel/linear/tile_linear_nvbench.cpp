#include "tile_linear.h"
#include <vector>
#include <torch/torch.h>
#include <nvbench/nvbench.cuh>
#include <cuda_runtime.h>

using namespace mllm;

void tile_linear_nvbench(nvbench::state& state) {
    state.collect_cupti_metrics();

    int M = static_cast<int>(state.get_int64("M"));
    int N = static_cast<int>(state.get_int64("N"));
    int K = static_cast<int>(state.get_int64("K"));

    torch::TensorOptions options = torch::dtype(torch::kHalf).device(torch::kCUDA);
    auto a = torch::randn({M, K}, options);
    auto b = torch::randn({N, K}, options);
    state.exec([&](nvbench::launch& launch){
        mllm::tile_linear(a, b);
    });

}

void linear_baseline_benchmark(nvbench::state& state) {
    state.collect_cupti_metrics();

    int M = static_cast<int>(state.get_int64("M"));
    int N = static_cast<int>(state.get_int64("N"));
    int K = static_cast<int>(state.get_int64("K"));

    torch::TensorOptions options = torch::dtype(torch::kHalf).device(torch::kCUDA);
    auto a = torch::randn({M, K}, options);
    auto b = torch::randn({N, K}, options);
    state.exec([&](nvbench::launch& launch){
        torch::nn::functional::linear(a, b);
    });

}


NVBENCH_BENCH(linear_baseline_benchmark)
    .add_int64_axis("M", {128})
    .add_int64_axis("N", {128})
    .add_int64_axis("K", {32});

NVBENCH_BENCH(tile_linear_nvbench)
    .add_int64_axis("M", {128})
    .add_int64_axis("N", {128})
    .add_int64_axis("K", {32});