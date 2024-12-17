#include "engine.h"
#include <gtest/gtest.h>
#include <glog/logging.h>

namespace mllm{

TEST(engine, add_requests_async){
    EngineConfig config;
    config.model_path = "/home/xzd/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/8c85e9a4d626b7b908448be32c1ba5ad79b95e76";

    Engine engine(config);

    BatchOutputCallback callback = [](size_t i, RequestOutput output) {
        std::cout << "callback is called" << std::endl;
        return true;
    };
    std::vector<std::string> prompts{"she", "he", "hello!", "he is"};
    std::vector<torch::Tensor> pixel_values{torch::randn({1, 3, 336, 336}), torch::randn({1, 3, 336, 336}), torch::randn({1, 3, 336, 336}), torch::randn({1, 3, 336, 336})};
    auto futures = engine.add_requests_async(prompts, pixel_values, false, callback);
    futures.wait();

    std::cout << "main thread finished" << std::endl;
    // std::this_thread::sleep_for(std::chrono::seconds(1));
    engine.stop();
}

TEST(engine, add_request_async){
    EngineConfig config;
    config.model_path = "/home/xzd/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/8c85e9a4d626b7b908448be32c1ba5ad79b95e76";

    Engine engine(config);

    std::function<bool(RequestOutput)> callback = [](RequestOutput output) {
        std::cout << "callback is called" << std::endl;
        return true;
    };
    engine.add_request_async("she", torch::randn({1, 3, 336, 336}), false, callback);
    engine.add_request_async("he", torch::randn({1, 3, 336, 336}), false, callback);
    engine.add_request_async("hello!", torch::randn({1, 3, 336, 336}), false, callback);
    auto future = engine.add_request_async("he is", torch::randn({1, 3, 336, 336}), false, callback);
    future.wait();

    std::cout << "main thread finished" << std::endl;
    // std::this_thread::sleep_for(std::chrono::seconds(1));
    engine.stop();
}

TEST(engine, engine_config){
    EngineConfig config;
    EXPECT_EQ(config.memory_config.block_size, 16);
}


}
