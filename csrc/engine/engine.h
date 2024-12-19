#pragma once
#include <folly/Function.h>

#include <vector>
#include <thread>
#include <future>
#include <atomic>
#include <torch/torch.h>

#include "memory/kv_cache.h"
#include "model/models.h"
#include "tokenizer/tokenizer.h"
#include "sequence.h"
#include "common/concurrent_queue.h"

namespace mllm{

struct MemoryConfig{
    enum class MemoryManagementPolicy{VANILLA = 0, SHARED = 1};
    MemoryManagementPolicy memory_management_policy = MemoryManagementPolicy::VANILLA;
    int num_blocks = 10000;
    int block_size = 16;
};
struct StageConfig{
    int default_max_tokens = 64;
    bool disaggregate_embed_prefill = true;
    enum class KVcacheEvictionPolicy{NONE = 0, RANDOM = 1, STREAMINGLLM = 2};
    KVcacheEvictionPolicy kv_cache_eviction_policy = KVcacheEvictionPolicy::NONE;
    int window_size = 28;
    int attention_sink_size = 4;
    enum class TokenPruningPolicy{NONE = 0, RANDOM = 1, STREAMINGLLM = 2};
    TokenPruningPolicy token_pruning_policy = TokenPruningPolicy::NONE;
    int n_embed_output_tokens = 64;
    bool debug_mode = false;
};
struct SchedulerConfig{
    enum class BatchPolicy{NOBATCH = 0, REQUESTLEVEL = 1, CONTINUOUSBATCH=2};
    BatchPolicy batch_policy = BatchPolicy::CONTINUOUSBATCH;
    enum class BatchPriority{PREFILL = 0, DECODE = 1};
    BatchPriority priority = BatchPriority::PREFILL;
    int max_running_sequences = 10;
    int max_batch_fill_tokens = 1024;
    int max_batch_embed_images = 3;
    bool batch_embed_fill = true;
    bool debug_mode = false;
};
struct EngineConfig{
    int num_handling_threads = 32;
    std::string model_path = "";
    MemoryConfig memory_config;
    StageConfig stage_config;
    SchedulerConfig scheduler_config;
    bool batch_image_embed_forward = true;
};
class SequenceScheduler{
public:
    SequenceScheduler(const SchedulerConfig& config);
    void schedule_new(Sequence* seq);
    void schedule_running(Sequence* seq);
    void step(std::vector<Sequence*>& this_step);
private:
    SchedulerConfig config;
    ConcurrentQueue<Sequence*> waiting;
    std::vector<Sequence*> running;
    std::vector<Sequence*> finished;
    int step_cnt = 0;
};


class BatchFuture {
public:
    BatchFuture(std::unique_ptr<std::vector<std::future<bool>>> futures) : futures_(std::move(futures)) {}

    void wait() {
        for (const auto& future : *futures_) future.wait();
    }

    std::vector<bool> get() {
        std::vector<bool> results;
        results.reserve(futures_->size());
        for (auto& future : *futures_) {
            results.push_back(future.get());
        }
        return results;
    }

private:
    // use unique_ptr to avoid static assertion failed:
    //  'result type must be constructible from input type'
    std::unique_ptr<std::vector<std::future<bool>>> futures_;
};

class Engine{
public:
    Engine(const EngineConfig& config);

private:
    std::future<bool> add_request(std::string prompt,
        torch::Tensor pixel_value, 
        const SamplingParams& sp, 
        bool stream,
        OutputCallback callback);

public:
    int step();
    void run_until_complete();
    std::future<bool> add_request_async(std::string prompt,
        torch::Tensor pixel_value, 
        const SamplingParams& sp, 
        bool stream,
        OutputCallback callback);

    BatchFuture add_requests_async(
        std::vector<std::string> prompts, 
        std::vector<torch::Tensor> pixel_values, 
        std::vector<SamplingParams> sps, 
        bool stream,
        BatchOutputCallback callback
        );

    void stop();
private:
    using Task = folly::Function<void(size_t tid)>;
    void handling_loop(size_t tid);
    ConcurrentQueue<Task> queue_;
    std::vector<std::thread> handling_threads_;
    std::vector<std::unique_ptr<Tokenizer>> tokenizers_;
    std::unique_ptr<Tokenizer> tokenizer;
    std::thread loop_thread_;
    std::atomic_bool stoped_{false};
    std::atomic_bool running_{false};
    torch::TensorOptions options_;
    std::vector<std::unique_ptr<KVCache>> kv_caches;
    EngineConfig config;
    // FakeConfig model_config;
    // FakeModel model{nullptr};
    LlavaConfig model_config;
    LlavaForConditionalGeneration model{nullptr};
    std::unique_ptr<SequenceScheduler> scheduler;
    void execute_batch_fill(std::vector<Sequence*>& seqs);
    void execute_batch_image_embed(std::vector<Sequence*>& seqs);
    void init_model();
};


}