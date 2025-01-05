#include "linear.h"
#include <vector>
#include <torch/torch.h>
#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

namespace mllm{

std::vector<std::tuple<std::string, linear_func>> kernels{
    {"linear_naive", mllm::linear_naive}, 
    {"linear_smem", linear_smem}, 
    {"linear_smem_v2", linear_smem_v2}
};

static void BM_linear_kernel(benchmark::State& state){
    if(!torch::cuda::is_available()){
        state.SkipWithMessage("CUDA is not availabe");
        return;
    }

    auto [kernel_name, kernel] = kernels[state.range(0)];
    torch::ScalarType dtype = static_cast<torch::ScalarType>(state.range(1));
    int n_tokens = static_cast<int>(state.range(2));
    int hidden_size = static_cast<int>(state.range(3));

    torch::TensorOptions options = torch::dtype(torch::kHalf).device(torch::kCUDA);
    auto h = torch::randn({n_tokens, hidden_size}, options);
    auto w = torch::randn({hidden_size, hidden_size}, options);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for(auto _ : state){
        cudaEventRecord(start);

        auto o = kernel(h, w);
        benchmark::DoNotOptimize(o);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        state.PauseTiming();
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds / 1000);
        state.ResumeTiming();
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    state.SetLabel(kernel_name + " " + torch::toString(dtype));
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

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    state.SetLabel(std::string("pytorch linear ") + torch::toString(dtype));
}

const std::vector<long int> dtypes = {
    static_cast<long int>(torch::kHalf)
};

BENCHMARK(BM_linear_kernel)
    ->ArgsProduct({{0, 1, 2}, dtypes, {4096}, {4096}});

BENCHMARK(BM_linear_baseline)
    ->ArgsProduct({dtypes, {4096}, {4096}});

}