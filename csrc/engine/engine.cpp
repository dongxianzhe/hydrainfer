#include "engine.h"
#include <iostream>
#include <filesystem>
#include "tokenizer/hf_tokenizer.h"
#include "model/fake_model.h"
#include "memory/kv_cache.h"

namespace mllm{

SequenceScheduler::SequenceScheduler(const SchedulerConfig& config) : config(config) {

}

void SequenceScheduler::schedule_new(Sequence* seq){
    waiting.push(seq);
}

void SequenceScheduler::schedule_running(const std::vector<Sequence*>& seqs){
    for(int i = 0;i < seqs.size();i ++)running.push_back(seqs[i]);
}

void SequenceScheduler::schedule_unfinished(const std::vector<Sequence*>& seqs){
    for(int i = 0;i < seqs.size();i ++)running.push_back(seqs[i]);
}

void SequenceScheduler::step(std::vector<Sequence*>& this_step){
    // empty_stages = 0, TextFill = 1, ImageFill = 2, ImageEmbedFill = 3, ImageEmbed = 4
    step_cnt += 1;
    if(config.batch_policy == SchedulerConfig::BatchPolicy::CONTINUOUSBATCH){
        while(running.size() < config.max_running_sequences && !waiting.empty())running.push_back(waiting.pop());
    }
    else if(config.batch_policy == SchedulerConfig::BatchPolicy::REQUESTLEVEL && running.size() == 0){
        while(running.size() < config.max_running_sequences && !waiting.empty())running.push_back(waiting.pop());
    }
    else if(config.batch_policy == SchedulerConfig::BatchPolicy::NOBATCH && running.size() == 0){
        if(!waiting.empty())running.push_back(waiting.pop());
    }
    int batch_fill_tokens = 0;
    int batch_embed_images = 0;

    std::vector<Sequence*> prefill_seqs;
    std::vector<Sequence*> decode_seqs;
    std::vector<Sequence*> embed_seqs;
    std::vector<Sequence*> next_step;
    for(int i = 0;i < running.size();i ++){
        Sequence* seq = running[i];
        Stage* stage = &(seq->stages[seq->curr_stage_id]);
        if(stage->type == Stage::StageType::TextFill || stage->type == Stage::StageType::ImageFill || stage->type == Stage::StageType::ImageEmbedFill){
            if(stage->token_ids.size() == 1)decode_seqs.push_back(seq);
            else prefill_seqs.push_back(seq);
        }
        else if(stage->type == Stage::StageType::ImageEmbed){
            embed_seqs.push_back(seq);
        }
        else{
            this_step.push_back(seq);
        }
    }
    if(prefill_seqs.size() > 0 && ! config.batch_embed_fill){
        for(int i = 0;i < embed_seqs.size();i ++)next_step.push_back(embed_seqs[i]);
    }
    else{
        for(int i = 0;i < embed_seqs.size();i ++){
            if(batch_embed_images < config.max_batch_embed_images){
                this_step.push_back(embed_seqs[i]);
                batch_embed_images += 1;
            }
            else{
                next_step.push_back(embed_seqs[i]);
            }
        }
    }

    std::vector<Sequence*>* first;
    std::vector<Sequence*>* second;
    if(config.priority == SchedulerConfig::BatchPriority::PREFILL){
        first = &prefill_seqs;
        second = &decode_seqs;
    }
    else{
        first = &decode_seqs;
        second = &prefill_seqs;
    }
    for(int i = 0;i < first->size();i ++){
        Sequence* seq = (*first)[i];
        Stage* stage = &(seq->stages[seq->curr_stage_id]);
        if(batch_fill_tokens < config.max_batch_fill_tokens){
            this_step.push_back(seq);
            batch_fill_tokens += stage->token_ids.size();
        }else{
            next_step.push_back(seq);
            batch_fill_tokens += stage->token_ids.size();
        }
    }
    for(int i = 0;i < second->size();i ++){
        Sequence* seq = (*second)[i];
        Stage* stage = &(seq->stages[seq->curr_stage_id]);
        if(batch_fill_tokens < config.max_batch_fill_tokens){
            this_step.push_back(seq);
            batch_fill_tokens += stage->token_ids.size();
        }else{
            next_step.push_back(seq);
            batch_fill_tokens += stage->token_ids.size();
        }
    }
    if(config.debug_mode){
        printf("------------------------------ step {%d} ------------------------------\n", step_cnt);
    }
    running = next_step;
}

Engine::Engine(const EngineConfig& config){
    // 1. model
    options_ = torch::dtype(torch::kHalf).device(torch::kCUDA);
    const std::string tokenizer_path = config.model_path + "/tokenizer.json";
    CHECK(std::filesystem::exists(tokenizer_path));
    auto tokenizer = HFTokenizer::from_file(tokenizer_path);

    FakeModel model(model_config, options_);
    // 2. memory
    for(int i = 0;i < model_config.n_layers;i ++){
        kv_caches.emplace_back(std::make_unique<KVCache>(config.memory_config.num_blocks / model_config.n_layers, config.memory_config.block_size, model_config.n_kv_heads, model_config.head_size, options_));
    }
    
    // 3. sequence build
    for(int i = 0;i < config.num_handling_threads;i ++){
        tokenizers_.emplace_back(tokenizer->clone());
        handling_threads_.emplace_back([this, i] {handling_loop(i);});
    }
}

void Engine::handling_loop(size_t tid){
    while(true){
        Task task = queue_.pop();
        if(task == nullptr)break;
        task(tid);
    }
}

std::future<bool> Engine::add_request(std::string prompt, torch::Tensor pixel_value, bool stream, OutputCallback callback){
    std::promise<bool> promise;
    auto future = promise.get_future();

    queue_.push([this, promise = std::move(promise), prompt = std::move(prompt), pixel_value=std::move(pixel_value), stream, callback = std::move(callback)](size_t tid) mutable {
        CHECK(!prompt.empty());
        std::cout << "todo process request and schedule sequence:" << prompt << std::endl;
        std::vector<int> token_ids;
        CHECK(tokenizers_[tid]->encode(prompt, &token_ids));
        for(int i = 0;i < token_ids.size();i ++)printf("%d ", token_ids[i]);puts("");
        promise.set_value(true);
    });

    return future;
}

std::future<bool> Engine::add_request_async(std::string prompt,
    torch::Tensor pixel_value, 
    bool stream,
    OutputCallback callback){

    return add_request(
        std::move(prompt), 
        std::move(pixel_value),
        stream, 
        [callback = std::move(callback)](const RequestOutput& output){
            return callback(output);
        }
    );
}

BatchFuture Engine::add_requests_async(
    std::vector<std::string> prompts,
    std::vector<torch::Tensor> pixel_values, 
    bool stream,
    BatchOutputCallback callback){

    CHECK(prompts.size() == pixel_values.size());
    int n_requests = prompts.size();
    auto futures = std::make_unique<std::vector<std::future<bool>>>();
    futures->reserve(n_requests);
    for(int i = 0;i < n_requests;i ++){
        auto future = add_request(
            std::move(prompts[i]), 
            std::move(pixel_values[i]),
            stream, 
            [i, callback](const RequestOutput& output){
                return callback(i, output);
            }
        );
        futures->emplace_back(std::move(future));    
    }
    return {std::move(futures)}; // why must use std::move?
}

void Engine::stop(){
    for(int i = 0;i < handling_threads_.size();i ++)queue_.push(nullptr);
    for(auto& thread : handling_threads_)thread.join();
}

void Engine::step(){
    // 1. schedule sequence
    std::vector<Sequence*> this_step;
    scheduler->step(this_step);
    // 2. batch sequence and execute
    // 3. schedule sequence
}


}

