#include "engine.h"
#include <iostream>

namespace mllm{


Engine::Engine(const EngineConfig& config){
    for(int i = 0;i < config.num_handling_threads;i ++){
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
        std::cout << "todo process request and schedule sequence:" << prompt << std::endl;
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

}

