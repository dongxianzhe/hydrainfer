#include <filesystem>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "hf_tokenizer.h"

TEST(hf_tokenizer, encode){
    using namespace mllm;
    std::string model_weights_path = "/home/xzd/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/8c85e9a4d626b7b908448be32c1ba5ad79b95e76";

    const std::string tokenizer_path = model_weights_path + "/tokenizer.json";
    EXPECT_TRUE(std::filesystem::exists(tokenizer_path));

    auto tokenizer = HFTokenizer::from_file(tokenizer_path);
    std::vector<int> token_ids;
    std::string prompt = "USER: <image>\nWhat is the content of this image?\nASSISTANT:";
    tokenizer->encode(prompt, &token_ids);
    for(int i = 0;i < token_ids.size();i ++)printf("%d ",token_ids[i]);puts("");
    printf("n_tokens %d\n", static_cast<int>(token_ids.size()));

    EXPECT_EQ(static_cast<int>(token_ids.size()), 21);
    std::vector<int> token_ids_ref{1, 3148, 1001, 29901, 29871, 32000, 29871, 13, 5618, 338, 278, 2793, 310, 445, 1967, 29973, 13, 22933, 9047, 13566, 29901};
    for(int i = 0;i < token_ids.size();i ++){
        EXPECT_EQ(token_ids[i], token_ids_ref[i]);
    }
}

TEST(hf_tokenizer, decode){
    using namespace mllm;
    std::string model_weights_path = "/home/xzd/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/8c85e9a4d626b7b908448be32c1ba5ad79b95e76";

    const std::string tokenizer_path = model_weights_path + "/tokenizer.json";
    EXPECT_TRUE(std::filesystem::exists(tokenizer_path));

    auto tokenizer = HFTokenizer::from_file(tokenizer_path);
    std::vector<int> token_ids;
    std::string prompt = "USER: <image>\nWhat is the content of this image?\nASSISTANT:";
    tokenizer->encode(prompt, &token_ids);
    for(int i = 0;i < token_ids.size();i ++)printf("%d ",token_ids[i]);puts("");
    printf("n_tokens %d\n", static_cast<int>(token_ids.size()));


    std::string output_text = tokenizer->decode(token_ids, false);
    std::cout << "output_text: " << output_text << std::endl;

    for(int i = 0;i < token_ids.size();i ++){
        std::cout << "id_to_token:" << token_ids[i] << " -> " << tokenizer->id_to_token(token_ids[i]) << std::endl;
    }

    for(int i = 0;i < token_ids.size();i ++){
        std::cout << "decode:" << tokenizer->decode(Slice<int>(token_ids).slice(i, i + 1), false) << std::endl;
    }
}