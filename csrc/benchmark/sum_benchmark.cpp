#include <benchmark/benchmark.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

extern void sum(at::Tensor& out, const at::Tensor& in);

static void BM_sum_baseline(benchmark::State& state, const torch::Device& device) {
    // Perform setup here
    torch::ScalarType dtype = static_cast<torch::ScalarType>(state.range(0));
    int n_elements = state.range(0);
    auto input = torch::randn({n_elements}, torch::dtype(dtype).device(device));
    for (auto _ : state) {
        // Call the implementation function
        auto output = torch::sum(input, /*dim*/ -1, /*keepdim*/ false);
        // don't optimize out the output
        benchmark::DoNotOptimize(output);
    }
    state.SetLabel(torch::toString(dtype));
}

static void BM_sum_kernel(benchmark::State& state) {
    // skip if no gpu
    if (!torch::cuda::is_available()) {
        state.SkipWithMessage("CUDA is not available");
        return;
    }

    // Perform setup here
    torch::ScalarType dtype = static_cast<torch::ScalarType>(state.range(0));
    int n_elements = state.range(1);
    auto input = torch::randn({n_elements}, torch::dtype(dtype).device(torch::kCUDA));
    auto output = torch::rand({1}, torch::dtype(dtype).device(torch::kCUDA));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (auto _ : state) {
        // Start measuring time
        cudaEventRecord(start);

        // Launch the CUDA kernel
        sum(output, input);
        // don't optimize out the output
        benchmark::DoNotOptimize(output);

        // Stop measuring time and calculate the elapsed time
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        state.PauseTiming();

        // Update the benchmark state with the measured time
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds / 1000);
        state.ResumeTiming();
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::string label = std::string("sum") + " " + torch::toString(dtype);
    state.SetLabel(label);
}

const std::vector<int64_t> dtypes = {static_cast<int64_t>(torch::kFloat),
                                     static_cast<int64_t>(torch::kHalf),
                                     };

// benchmark for kernels
BENCHMARK(BM_sum_kernel)
    ->ArgsProduct({dtypes, {4096, 4097, 20560}});

// benchmark for gpus
BENCHMARK_CAPTURE(BM_sum_baseline, "gpu", torch::kCUDA)
    ->ArgsProduct({dtypes, {4096, 4097, 20560}});

// benchmark for cpus
BENCHMARK_CAPTURE(BM_sum_baseline, "cpu", torch::kCPU)
    ->ArgsProduct({dtypes, {4097, 4097, 20560}});