#include "linear.h"
#include <vector>
#include <torch/torch.h>
#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

namespace mllm{

static void BM_linear_kernel(benchmark::State& state){
    if(!torch::cuda::is_available()){
        state.SkipWithMessage("CUDA is not availabe");
        return;
    }
    torch::ScalarType dtype = static_cast<torch::ScalarType>(state.range(0));
    int n_tokens = static_cast<int>(state.range(1));
    int hidden_size = static_cast<int>(state.range(2));

    torch::TensorOptions options = torch::dtype(torch::kHalf).device(torch::kCUDA);
    auto h = torch::randn({n_tokens, hidden_size}, options);
    auto w = torch::randn({hidden_size, hidden_size}, options);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for(auto _ : state){
        cudaEventRecord(start);

        auto o = mllm::linear(h, w);
        benchmark::DoNotOptimize(o);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        state.PauseTiming();
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds / 1000);
        state.ResumeTiming();
    }
}

static void BM_linear_baseline(benchmark::State& state){
    if(!torch::cuda::is_available()){
        state.SkipWithMessage("CUDA is not availabe");
        return;
    }
    torch::ScalarType dtype = static_cast<torch::ScalarType>(state.range(0));
    int n_tokens = static_cast<int>(state.range(1));
    int hidden_size = static_cast<int>(state.range(2));

    torch::TensorOptions options = torch::dtype(torch::kHalf).device(torch::kCUDA);
    auto h = torch::randn({n_tokens, hidden_size}, options);
    auto w = torch::randn({hidden_size, hidden_size}, options);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for(auto _ : state){
        cudaEventRecord(start);

        auto o = torch::nn::functional::linear(h, w);
        benchmark::DoNotOptimize(o);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        state.PauseTiming();
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds / 1000);
        state.ResumeTiming();
    }
}

const std::vector<long int> dtypes = {
    static_cast<long int>(torch::kHalf)
};

BENCHMARK(BM_linear_kernel)
    ->ArgsProduct({dtypes, {4096}, {4096}});

BENCHMARK(BM_linear_baseline)
    ->ArgsProduct({dtypes, {4096}, {4096}});

}