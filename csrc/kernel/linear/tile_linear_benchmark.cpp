#include "tile_linear.h"
#include <vector>
#include <torch/torch.h>
#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

namespace mllm{

std::vector<std::tuple<std::string, tile_linear_func>> kernels{
    {"tile_linear", mllm::tile_linear}, 
};

static void BM_tile_linear_kernel(benchmark::State& state){
    if(!torch::cuda::is_available()){
        state.SkipWithMessage("CUDA is not availabe");
        return;
    }

    auto [kernel_name, kernel] = kernels[state.range(0)];
    torch::ScalarType dtype = static_cast<torch::ScalarType>(state.range(1));
    int M = static_cast<int>(state.range(2));
    int N = static_cast<int>(state.range(3));
    int K = static_cast<int>(state.range(4));

    torch::TensorOptions options = torch::dtype(torch::kHalf).device(torch::kCUDA);
    auto a = torch::randn({M, K}, options);
    auto b = torch::randn({N, K}, options);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for(auto _ : state){
        cudaEventRecord(start);

        auto o = kernel(a, b);
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

static void BM_tile_linear_baseline(benchmark::State& state){
    if(!torch::cuda::is_available()){
        state.SkipWithMessage("CUDA is not availabe");
        return;
    }
    torch::ScalarType dtype = static_cast<torch::ScalarType>(state.range(0));
    int M = static_cast<int>(state.range(1));
    int N = static_cast<int>(state.range(2));
    int K = static_cast<int>(state.range(3));

    torch::TensorOptions options = torch::dtype(torch::kHalf).device(torch::kCUDA);
    auto a = torch::randn({M, K}, options);
    auto b = torch::randn({N, K}, options);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for(auto _ : state){
        cudaEventRecord(start);

        auto o = torch::nn::functional::linear(a, b);
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

BENCHMARK(BM_tile_linear_kernel)
    ->ArgsProduct({{0}, dtypes, {128}, {128}, {32}});

BENCHMARK(BM_tile_linear_baseline)
    ->ArgsProduct({dtypes, {128}, {128} , {32}});

}