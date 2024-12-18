#include "state_dict.h"

#include <filesystem>
#include <c10/core/Device.h>
#include <gtest/gtest.h>

namespace mllm {

TEST(StateDictTest, load_llava_safetensor) {
  std::string model_weights_path = "/home/xzd/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/8c85e9a4d626b7b908448be32c1ba5ad79b95e76";
  std::vector<std::string> model_weights_files;
  for (const auto& entry : std::filesystem::directory_iterator(model_weights_path)) {
      if (entry.path().extension() == ".safetensors") {
          model_weights_files.push_back(entry.path().string());
      }
  }
  for(int i = 0;i < model_weights_files.size();i ++){
      std::cout << model_weights_files[i] << std::endl;
  }

  int total_tensor = 0;
  for(int i = 0;i < model_weights_files.size();i ++){
    auto state_dict = StateDict::load_safetensors(model_weights_files[i]);
    total_tensor += state_dict->size();

    for(auto it = state_dict->begin(); it != state_dict->end(); it ++){
      std::string key = it->first;
      torch::Tensor weight = it->second;
      std::cout << key << ": " << weight.sizes() << std::endl;
    }
  }
  std::cout << "total_tensor: " << total_tensor << std::endl;
  EXPECT_EQ(total_tensor, 686);
}

}  // namespace llm
