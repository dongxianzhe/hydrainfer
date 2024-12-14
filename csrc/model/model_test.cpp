#include <iostream>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <c10/core/TensorOptions.h>
#include "memory/kv_cache.h"
#include "llama.h"


namespace mllm{

//     @classmethod
//     def from_safetensor(cls, model_weights_path: str, dtype: torch.dtype, device: torch.device):
//         # 1. create model
//         config = LlamaConfig.from_pretrained(model_weights_path)
//         torch.set_default_dtype(dtype)
//         with torch.device(device):
//             model = cls(config)
//         torch.set_default_dtype(torch.float)

//         # 2. load weights
//         state_dict = model.state_dict()
//         loaded_set = set()
//         for entry in os.scandir(model_weights_path):
//             if entry.is_file() and os.path.splitext(entry.name)[1] == '.safetensors':
//                 print(f'load safetensor from {entry.path}')
//                 for name, weight in safetensors.torch.load_file(entry.path).items():
//                     if name.endswith('.self_attn.rotary_emb.inv_freq'):
//                         continue
//                     state_dict[name].data.copy_(weight)
//                     loaded_set.add(name)
//         model.load_state_dict(state_dict)
//         model.to(dtype)
//         model.eval()

//         # 3. verify
//         assert len(loaded_set) == len(state_dict)

//         return model

TEST(model, model){
    std::vector<int> q_seq_lens{1, 15, 111, 1};
    std::vector<int> k_seq_lens{100, 15, 234, 1024};
    // std::vector<int> q_seq_lens{1};
    // std::vector<int> k_seq_lens{100};
    int n_qo_heads = 8;
    int n_kv_heads = 8;
    int head_size = 128;
    int n_blocks = 10000;
    int block_size = 16;
    torch::Dtype dtype(torch::kHalf);
    torch::Device device(torch::kCUDA);
    torch::TensorOptions options = torch::dtype(dtype).device(device);
    torch::TensorOptions params_options = torch::dtype(torch::kInt32).device(device);
    int n_tokens = 0;
    for(int i = 0;i < q_seq_lens.size();i ++)n_tokens += q_seq_lens[i];
    int n_kv_cache_tokens = 0;
    for(int i = 0;i < k_seq_lens.size();i ++)n_kv_cache_tokens += k_seq_lens[i];
    KVCache kv_cache(n_blocks, block_size, n_kv_heads, head_size, torch::kHalf, torch::kCUDA);
    std::vector<int> new_cache_slots;
    std::vector<int> block_tables;
    std::vector<int> cu_block_lens{0};
    int num_sequences = q_seq_lens.size();
    for(int i = 0;i < num_sequences;i ++){
        int q_seq_len = q_seq_lens[i];
        int k_seq_len = k_seq_lens[i];
        int n_blocks_allocate = (k_seq_len + block_size - 1) / block_size;
        std::vector<int> block_table = kv_cache.allocate(n_blocks_allocate);
        for(int position_id = k_seq_len - q_seq_len; position_id < k_seq_len; position_id ++){
            int block_id = block_table[position_id / block_size];
            int block_offset = position_id % block_size;
            int slot_id = block_id * block_size + block_offset;
            new_cache_slots.push_back(slot_id);
        }
        for(int block_id : block_table)block_tables.push_back(block_id);
        cu_block_lens.push_back(cu_block_lens.back() + n_blocks_allocate);
    }
    std::vector<int> q_cu_seq_lens{0};
    for(int i = 0;i < q_seq_lens.size();i ++)q_cu_seq_lens.push_back(q_cu_seq_lens.back() + q_seq_lens[i]);
    std::vector<int> k_cu_seq_lens{0};
    for(int i = 0;i < k_seq_lens.size();i ++)k_cu_seq_lens.push_back(k_cu_seq_lens.back() + k_seq_lens[i]);

    printf("q_cu_seq_lens: ");for(int i = 0;i < q_cu_seq_lens.size();i ++)printf("%d ",q_cu_seq_lens[i]);puts("");
    printf("k_cu_seq_lens: ");for(int i = 0;i < k_cu_seq_lens.size();i ++)printf("%d ",k_cu_seq_lens[i]);puts("");
    printf("new_cache_slots: ");for(int i = 0;i < new_cache_slots.size();i ++)printf("%d ", new_cache_slots[i]);puts("");
    printf("block_tables: ");for(int i = 0;i < block_tables.size();i ++)printf("%d ", block_tables[i]);puts("");
    printf("cu_block_lens: ");for(int i = 0;i < cu_block_lens.size();i ++)printf("%d ", cu_block_lens[i]);puts("");

    AttentionParams params;
    params.kv_cache = &kv_cache;
    params.q_cu_seq_lens = torch::tensor(q_cu_seq_lens, params_options);
    params.k_cu_seq_lens = torch::tensor(k_cu_seq_lens, params_options);
    params.new_cache_slots = torch::tensor(new_cache_slots, params_options);
    params.block_tables = torch::tensor(block_tables, params_options);
    params.cu_block_lens = torch::tensor(cu_block_lens, params_options);
    params.num_sequences = num_sequences;
    params.all_sequences_decode = false;
    params.q_max_seq_len = 1024;
    params.k_max_seq_len = 1024;

    LlamaConfig config; 
    LlamaForCausalLM model(config, options);

    ModelParameters model_params;
    for(int i = 0;i < config.n_layers; i ++)model_params.attention_params.push_back(&params);
    // auto h = torch::randn({n_tokens, config.hidden_size}, torch::dtype(torch::kHalf).device(torch::kCUDA));
    auto input_ids = torch::arange(n_tokens, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto p = torch::arange({n_tokens}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    auto out = model(input_ids, p, model_params);

    std::cout << out.sizes() << std::endl;
}


// TEST(model, llamadecodelayer){
//     // std::vector<int> q_seq_lens{1, 15, 111, 1};
//     // std::vector<int> k_seq_lens{100, 15, 234, 1024};
//     std::vector<int> q_seq_lens{1};
//     std::vector<int> k_seq_lens{100};
//     int n_qo_heads = 8;
//     int n_kv_heads = 8;
//     int head_size = 128;
//     int n_blocks = 10000;
//     int block_size = 16;
//     torch::Dtype dtype(torch::kHalf);
//     torch::Device device(torch::kCUDA);
//     torch::TensorOptions options = torch::dtype(dtype).device(device);
//     torch::TensorOptions params_options = torch::dtype(torch::kInt32).device(device);
//     int n_tokens = 0;
//     for(int i = 0;i < q_seq_lens.size();i ++)n_tokens += q_seq_lens[i];
//     int n_kv_cache_tokens = 0;
//     for(int i = 0;i < k_seq_lens.size();i ++)n_kv_cache_tokens += k_seq_lens[i];
//     KVCache kv_cache(n_blocks, block_size, n_kv_heads, head_size, torch::kHalf, torch::kCUDA);
//     std::vector<int> new_cache_slots;
//     std::vector<int> block_tables;
//     std::vector<int> cu_block_lens{0};
//     int num_sequences = q_seq_lens.size();
//     for(int i = 0;i < num_sequences;i ++){
//         int q_seq_len = q_seq_lens[i];
//         int k_seq_len = k_seq_lens[i];
//         int n_blocks_allocate = (k_seq_len + block_size - 1) / block_size;
//         std::vector<int> block_table = kv_cache.allocate(n_blocks_allocate);
//         for(int position_id = k_seq_len - q_seq_len; position_id < k_seq_len; position_id ++){
//             int block_id = block_table[position_id / block_size];
//             int block_offset = position_id % block_size;
//             int slot_id = block_id * block_size + block_offset;
//             new_cache_slots.push_back(slot_id);
//         }
//         for(int block_id : block_table)block_tables.push_back(block_id);
//         cu_block_lens.push_back(cu_block_lens.back() + n_blocks_allocate);
//     }
//     std::vector<int> q_cu_seq_lens{0};
//     for(int i = 0;i < q_seq_lens.size();i ++)q_cu_seq_lens.push_back(q_cu_seq_lens.back() + q_seq_lens[i]);
//     std::vector<int> k_cu_seq_lens{0};
//     for(int i = 0;i < k_seq_lens.size();i ++)k_cu_seq_lens.push_back(k_cu_seq_lens.back() + k_seq_lens[i]);

//     printf("q_cu_seq_lens: ");for(int i = 0;i < q_cu_seq_lens.size();i ++)printf("%d ",q_cu_seq_lens[i]);puts("");
//     printf("k_cu_seq_lens: ");for(int i = 0;i < k_cu_seq_lens.size();i ++)printf("%d ",k_cu_seq_lens[i]);puts("");
//     printf("new_cache_slots: ");for(int i = 0;i < new_cache_slots.size();i ++)printf("%d ", new_cache_slots[i]);puts("");
//     printf("block_tables: ");for(int i = 0;i < block_tables.size();i ++)printf("%d ", block_tables[i]);puts("");
//     printf("cu_block_lens: ");for(int i = 0;i < cu_block_lens.size();i ++)printf("%d ", cu_block_lens[i]);puts("");

//     AttentionParams params;
//     params.kv_cache = &kv_cache;
//     params.q_cu_seq_lens = torch::tensor(q_cu_seq_lens, params_options);
//     params.k_cu_seq_lens = torch::tensor(k_cu_seq_lens, params_options);
//     params.new_cache_slots = torch::tensor(new_cache_slots, params_options);
//     params.block_tables = torch::tensor(block_tables, params_options);
//     params.cu_block_lens = torch::tensor(cu_block_lens, params_options);
//     params.num_sequences = num_sequences;
//     params.all_sequences_decode = false;
//     params.q_max_seq_len = 1024;
//     params.k_max_seq_len = 1024;

//     LlamaConfig config; 
//     LlamaDecoderLayer model(config, options, 0);

//     ModelParameters model_params;
//     for(int i = 0;i < config.n_layers; i ++)model_params.attention_params.push_back(&params);
//     auto h = torch::randn({n_tokens, config.hidden_size}, torch::dtype(torch::kHalf).device(torch::kCUDA));
//     auto p = torch::arange({n_tokens}, torch::dtype(torch::kInt32).device(torch::kCUDA));
//     auto out = model(h, p, model_params);

//     std::cout << out.sizes() << std::endl;
// }

// TEST(model, llamaattention){
//     // std::vector<int> q_seq_lens{1, 15, 111, 1};
//     // std::vector<int> k_seq_lens{100, 15, 234, 1024};
//     std::vector<int> q_seq_lens{1};
//     std::vector<int> k_seq_lens{100};
//     int n_qo_heads = 8;
//     int n_kv_heads = 8;
//     int head_size = 128;
//     int n_blocks = 10000;
//     int block_size = 16;
//     torch::Dtype dtype(torch::kHalf);
//     torch::Device device(torch::kCUDA);
//     torch::TensorOptions options = torch::dtype(dtype).device(device);
//     torch::TensorOptions params_options = torch::dtype(torch::kInt32).device(device);
//     int n_tokens = 0;
//     for(int i = 0;i < q_seq_lens.size();i ++)n_tokens += q_seq_lens[i];
//     int n_kv_cache_tokens = 0;
//     for(int i = 0;i < k_seq_lens.size();i ++)n_kv_cache_tokens += k_seq_lens[i];
//     KVCache kv_cache(n_blocks, block_size, n_kv_heads, head_size, torch::kHalf, torch::kCUDA);
//     std::vector<int> new_cache_slots;
//     std::vector<int> block_tables;
//     std::vector<int> cu_block_lens{0};
//     int num_sequences = q_seq_lens.size();
//     for(int i = 0;i < num_sequences;i ++){
//         int q_seq_len = q_seq_lens[i];
//         int k_seq_len = k_seq_lens[i];
//         int n_blocks_allocate = (k_seq_len + block_size - 1) / block_size;
//         std::vector<int> block_table = kv_cache.allocate(n_blocks_allocate);
//         for(int position_id = k_seq_len - q_seq_len; position_id < k_seq_len; position_id ++){
//             int block_id = block_table[position_id / block_size];
//             int block_offset = position_id % block_size;
//             int slot_id = block_id * block_size + block_offset;
//             new_cache_slots.push_back(slot_id);
//         }
//         for(int block_id : block_table)block_tables.push_back(block_id);
//         cu_block_lens.push_back(cu_block_lens.back() + n_blocks_allocate);
//     }
//     std::vector<int> q_cu_seq_lens{0};
//     for(int i = 0;i < q_seq_lens.size();i ++)q_cu_seq_lens.push_back(q_cu_seq_lens.back() + q_seq_lens[i]);
//     std::vector<int> k_cu_seq_lens{0};
//     for(int i = 0;i < k_seq_lens.size();i ++)k_cu_seq_lens.push_back(k_cu_seq_lens.back() + k_seq_lens[i]);

//     printf("q_cu_seq_lens: ");for(int i = 0;i < q_cu_seq_lens.size();i ++)printf("%d ",q_cu_seq_lens[i]);puts("");
//     printf("k_cu_seq_lens: ");for(int i = 0;i < k_cu_seq_lens.size();i ++)printf("%d ",k_cu_seq_lens[i]);puts("");
//     printf("new_cache_slots: ");for(int i = 0;i < new_cache_slots.size();i ++)printf("%d ", new_cache_slots[i]);puts("");
//     printf("block_tables: ");for(int i = 0;i < block_tables.size();i ++)printf("%d ", block_tables[i]);puts("");
//     printf("cu_block_lens: ");for(int i = 0;i < cu_block_lens.size();i ++)printf("%d ", cu_block_lens[i]);puts("");

//     AttentionParams params;
//     params.kv_cache = &kv_cache;
//     params.q_cu_seq_lens = torch::tensor(q_cu_seq_lens, params_options);
//     params.k_cu_seq_lens = torch::tensor(k_cu_seq_lens, params_options);
//     params.new_cache_slots = torch::tensor(new_cache_slots, params_options);
//     params.block_tables = torch::tensor(block_tables, params_options);
//     params.cu_block_lens = torch::tensor(cu_block_lens, params_options);
//     params.num_sequences = num_sequences;
//     params.all_sequences_decode = false;
//     params.q_max_seq_len = 1024;
//     params.k_max_seq_len = 1024;


//     LlamaConfig config; 
//     LlamaSdpaAttention model(config, options);
//     auto h = torch::randn({n_tokens, config.hidden_size}, torch::dtype(torch::kHalf).device(torch::kCUDA));
//     auto p = torch::arange({n_tokens}, torch::dtype(torch::kInt32).device(torch::kCUDA));
//     auto out = model(h, p, params);
//     std::cout << out.sizes() << std::endl;
// }

// TEST(model, linear){
//     torch::TensorOptions options = torch::dtype(torch::kHalf).device(torch::kCUDA);
//     Linear model(2, 1, true, options);
//     auto x = torch::randn({2}, options);
//     auto o = model(x);
//     std::cout << x << std::endl;
//     std::cout << model->weight_ << std::endl;
//     std::cout << model->bias_ << std::endl;
//     std::cout << o << std::endl;
// }


// TEST(model, rmsnorm){
//     torch::TensorOptions options = torch::dtype(torch::kHalf).device(torch::kCUDA);
//     LlamaRMSNorm model(4096, 1e-5, options);
//     auto h = torch::randn({10, 4096}, options);
//     auto o = model(h);
//     std::cout << o.sizes() << std::endl;
//     EXPECT_FALSE(torch::equal(o, h));
// }

// TEST(kernel, mha_varlen_fwd){
//     int n_blocks = 10000;
//     int block_size = 16;
//     int n_qo_heads = 8;
//     int n_kv_heads = 8;
//     int head_dim = 128;
//     float sm_scale_ = 1.0 / std::sqrt(head_dim);
//     torch::TensorOptions options = torch::dtype(torch::kHalf).device(torch::kCUDA);
//     KVCache kv_cache(n_blocks, block_size, n_kv_heads, head_dim, torch::kHalf, torch::kCUDA);
//     auto [key_cache, value_cache] = kv_cache.get_kv_cache();
//     // auto key_cache   = torch::randn({n_blocks, block_size, n_kv_heads, head_dim}, options);
//     // auto value_cache = torch::randn({n_blocks, block_size, n_kv_heads, head_dim}, options);
//     int n_tokens = 100;
//     torch::Tensor out = torch::randn({n_tokens, n_qo_heads, head_dim}, options);
//     torch::Tensor q   = torch::randn({n_tokens, n_qo_heads, head_dim}, options);

//     torch::TensorOptions params_options = torch::dtype(torch::kInt32).device(torch::kCUDA);
//     auto q_cu_seq_lens = torch::tensor({0, 1}, params_options);
//     auto k_cu_seq_lens = torch::tensor({0, 100}, params_options);
//     auto block_tables  = torch::tensor({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, params_options);
//     auto cu_block_lens = torch::tensor({0, 10}, params_options);
//     mha_varlen_fwd(
//         out,
//         q,
//         key_cache, 
//         value_cache, 
//         q_cu_seq_lens, 
//         k_cu_seq_lens, 
//         block_tables, 
//         cu_block_lens, 
//         torch::nullopt, 
//         128, 
//         128, 
//         sm_scale_, 
//         0,
//         -1,
//         0,
//         0
//     );
//     std::cout << out.sizes() << std::endl;
// }

// TEST(attention, forward){
//     // std::vector<int> q_seq_lens{1, 15, 111, 1};
//     // std::vector<int> k_seq_lens{100, 15, 234, 1024};

//     std::vector<int> q_seq_lens{1};
//     std::vector<int> k_seq_lens{100};
//     int n_qo_heads = 8;
//     int n_kv_heads = 8;
//     int head_size = 128;
//     int n_blocks = 10000;
//     int block_size = 16;
//     torch::Dtype dtype(torch::kHalf);
//     torch::Device device(torch::kCUDA);
//     torch::TensorOptions options = torch::dtype(dtype).device(device);
//     torch::TensorOptions params_options = torch::dtype(torch::kInt32).device(device);
//     int n_tokens = 0;
//     for(int i = 0;i < q_seq_lens.size();i ++)n_tokens += q_seq_lens[i];
//     int n_kv_cache_tokens = 0;
//     for(int i = 0;i < k_seq_lens.size();i ++)n_kv_cache_tokens += k_seq_lens[i];
//     auto query = torch::randn({n_tokens, head_size * n_qo_heads}, options);
//     auto key   = torch::randn({n_tokens, head_size * n_kv_heads}, options);
//     auto value = torch::randn({n_tokens, head_size * n_kv_heads}, options);
//     KVCache kv_cache(n_blocks, block_size, n_kv_heads, head_size, torch::kHalf, torch::kCUDA);
//     std::vector<int> new_cache_slots;
//     std::vector<int> block_tables;
//     std::vector<int> cu_block_lens{0};
//     int num_sequences = q_seq_lens.size();
//     for(int i = 0;i < num_sequences;i ++){
//         int q_seq_len = q_seq_lens[i];
//         int k_seq_len = k_seq_lens[i];
//         int n_blocks_allocate = (k_seq_len + block_size - 1) / block_size;
//         std::vector<int> block_table = kv_cache.allocate(n_blocks_allocate);
//         for(int position_id = k_seq_len - q_seq_len; position_id < k_seq_len; position_id ++){
//             int block_id = block_table[position_id / block_size];
//             int block_offset = position_id % block_size;
//             int slot_id = block_id * block_size + block_offset;
//             new_cache_slots.push_back(slot_id);
//         }
//         for(int block_id : block_table)block_tables.push_back(block_id);
//         cu_block_lens.push_back(cu_block_lens.back() + n_blocks_allocate);
//     }
//     std::vector<int> q_cu_seq_lens{0};
//     for(int i = 0;i < q_seq_lens.size();i ++)q_cu_seq_lens.push_back(q_cu_seq_lens.back() + q_seq_lens[i]);
//     std::vector<int> k_cu_seq_lens{0};
//     for(int i = 0;i < k_seq_lens.size();i ++)k_cu_seq_lens.push_back(k_cu_seq_lens.back() + k_seq_lens[i]);

//     printf("q_cu_seq_lens: ");for(int i = 0;i < q_cu_seq_lens.size();i ++)printf("%d ",q_cu_seq_lens[i]);puts("");
//     printf("k_cu_seq_lens: ");for(int i = 0;i < k_cu_seq_lens.size();i ++)printf("%d ",k_cu_seq_lens[i]);puts("");
//     printf("new_cache_slots: ");for(int i = 0;i < new_cache_slots.size();i ++)printf("%d ", new_cache_slots[i]);puts("");
//     printf("block_tables: ");for(int i = 0;i < block_tables.size();i ++)printf("%d ", block_tables[i]);puts("");
//     printf("cu_block_lens: ");for(int i = 0;i < cu_block_lens.size();i ++)printf("%d ", cu_block_lens[i]);puts("");

//     AttentionParams params;
//     params.kv_cache = &kv_cache;
//     params.q_cu_seq_lens = torch::tensor(q_cu_seq_lens, params_options);
//     params.k_cu_seq_lens = torch::tensor(k_cu_seq_lens, params_options);
//     params.new_cache_slots = torch::tensor(new_cache_slots, params_options);
//     params.block_tables = torch::tensor(block_tables, params_options);
//     params.cu_block_lens = torch::tensor(cu_block_lens, params_options);
//     params.num_sequences = num_sequences;
//     params.all_sequences_decode = false;
//     params.q_max_seq_len = 1024;
//     params.k_max_seq_len = 1024;

//     CausalGroupedQueryPageAttention model(n_qo_heads, n_kv_heads, head_size);
//     model->to(dtype);
//     model->to(device);
//     auto out = model(query, key, value, params);
//     std::cout << out.sizes() << std::endl;
// }

// TEST(model, llamamlp){
//     int hidden_size = 1024;
//     int intermediate_size = 4096;
//     LlamaConfig config{hidden_size, intermediate_size};
//     LlamaMLP model(config);
//     int n_tokens = 10;
//     auto x = torch::randn({n_tokens, hidden_size});
//     auto y = model(x);
//     std::cout << y.sizes() << std::endl;
// }

// TEST(model ,rotart_embeding){
//     int rotary_dim = 128;
//     int max_position_embeddings = 4096;
//     torch::Tensor inv_freq = compute_default_inv_freq(rotary_dim, 10000);
//     bool interleaved = false;
//     torch::TensorOptions options = torch::dtype(torch::kHalf).device(torch::kCUDA);
//     RotaryEmbedding model(
//         rotary_dim, 
//         max_position_embeddings, 
//         inv_freq, 
//         interleaved, 
//         options
//     );
//     int n_tokens = 10;
//     int num_kv_heads = 32;
//     int head_dim = 128;
//     auto query = torch::randn({n_tokens, num_kv_heads, head_dim}, options);
//     auto key = torch::randn({n_tokens, num_kv_heads, head_dim}, options);
//     auto postion_ids = torch::randperm(max_position_embeddings, torch::dtype(torch::kInt).device(torch::kCUDA)).slice(0, 0, n_tokens);
//     auto query_ref = query.clone();
//     auto key_ref = key.clone();
//     auto [query_out, key_out] = model->forward(query, key, postion_ids); 
//     EXPECT_TRUE(torch::equal(query_out, query));
//     EXPECT_TRUE(torch::equal(key_out, key));
//     EXPECT_FALSE(torch::equal(query_ref, query));
//     EXPECT_FALSE(torch::equal(key_ref, query));
// }


// TEST(t, unsequeeze){
//     auto a = torch::arange(3);
//     auto sliced = a.index({torch::indexing::None, torch::indexing::Slice(), torch::indexing::None});
//     std::cout << a.sizes() << std::endl;
//     std::cout << sliced.sizes() << std::endl;
// }

// TEST(t, cat){
//     auto a = torch::zeros({2, 3}, torch::kFloat32);
//     auto b = torch::ones({2, 3}, torch::kFloat32);
//     auto c = torch::cat({a, b}, 0);
//     auto d = torch::cat({a, b}, 1);
//     std::cout << c.sizes() << std::endl;
//     std::cout << d.sizes() << std::endl;
//     std::cout << c << std::endl;
//     std::cout << d << std::endl;
// }

// TEST(t, cos){
//     auto x = torch::arange(3, torch::kFloat32);
//     auto y = x.cos();
//     std::cout << y << std::endl;
// }

// TEST(t, einsum){
//     auto a = torch::arange(3);
//     auto b = torch::arange(3);
//     auto c = torch::einsum("i, j->ij", {a, b});
//     std::cout << c << std::endl;
//     //     freqs = torch.einsum("i,j->ij", t, self.inv_freq) # (max_position_embedding, rotary_dim / 2)
// }

// TEST(t, arange){
//     auto t = torch::arange(10, torch::dtype(torch::kInt).device(torch::kCUDA));
//     std::cout << t << std::endl;
// }

// TEST(model, creation){
//     torch::Dtype dtype(torch::kHalf);
//     torch::Device device(torch::kCUDA);
//     const torch::TensorOptions& options = torch::dtype(dtype).device(device);
//     MLP mlp(1024, 1024);
//     mlp->to(dtype);
//     mlp->to(device);
//     torch::Tensor input = torch::randn({32, 1024}, options);
//     auto output = mlp(input);
//     auto output_ref = mlp->activation_fn(mlp->c_fc_(input));
//     EXPECT_TRUE(torch::equal(output, output_ref));
// }


// TEST(t, linear){
//     torch::Dtype dtype(torch::kHalf);
//     torch::Device device(torch::kCUDA);
//     const torch::TensorOptions& options = torch::dtype(dtype).device(device);

//     torch::nn::Linear linear(1024, 1024);
//     // linear->to(dtype);
//     // linear->to(device);

//     at::Tensor input = torch::randn({32, 1024}, options);
//     at::Tensor output = linear(input);
//     std::cout << "Output shape: " << output.sizes() << std::endl;
// }
}