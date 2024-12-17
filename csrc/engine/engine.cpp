#include "engine.h"
#include <iostream>
#include <filesystem>
#include "tokenizer/hf_tokenizer.h"
#include "model/fake_model.h"
#include "memory/kv_cache.h"
#include "model/model_params.h"

namespace mllm{

SequenceScheduler::SequenceScheduler(const SchedulerConfig& config) : config(config) {

}

void SequenceScheduler::schedule_new(Sequence* seq){
    waiting.push(seq);
}

void SequenceScheduler::schedule_running(Sequence* seq){
    running.push_back(seq);
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

Engine::Engine(const EngineConfig& config) : config(config) {
    // 1. model
    options_ = torch::dtype(torch::kHalf).device(torch::kCUDA);
    const std::string tokenizer_path = config.model_path + "/tokenizer.json";
    CHECK(std::filesystem::exists(tokenizer_path));
    auto tokenizer = HFTokenizer::from_file(tokenizer_path);

    model = FakeModel(model_config, options_);
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

std::future<bool> Engine::add_request(std::string prompt, torch::Tensor pixel_value, const SamplingParams& sp, bool stream, OutputCallback callback){
    std::promise<bool> promise;
    auto future = promise.get_future();

    queue_.push([this, promise = std::move(promise), prompt = std::move(prompt), pixel_value=std::move(pixel_value), sp = std::move(sp), stream, callback = std::move(callback)](size_t tid) mutable {
        CHECK(!prompt.empty());
        // 1. tokenize
        std::vector<int> token_ids;
        CHECK(tokenizers_[tid]->encode(prompt, &token_ids));
        std::vector<int> inserted_token_ids;
        inserted_token_ids.reserve(token_ids.size() + model_config.n_image_tokens - 1);
        for(int i = 0;i < token_ids.size();i ++){
            if(token_ids[i] != model_config.image_token_id)inserted_token_ids.push_back(token_ids[i]);
            else for(int i = 0;i < model_config.n_image_tokens;i ++)inserted_token_ids.push_back(token_ids[i]);
        }
        Sequence* seq = new Sequence();
        // 2. chunked stages
        if(config.stage_config.disaggregate_embed_prefill){
            Stage image_embed(Stage::StageType::ImageEmbed);
            Stage prefill(Stage::StageType::TextFill);
            seq->stages.push_back(image_embed);
            seq->stages.push_back(prefill);
            image_embed.pixel_values = pixel_value;
            image_embed.image_feature_dst_stage_id = 1;
            prefill.token_ids = token_ids;
            for(int i = 0;i < token_ids.size();i ++)prefill.position_ids.push_back(i);
            for(int i = 0;i < token_ids.size();i ++)prefill.cache_ids.push_back(i);
            prefill.sample = true;
            prefill.sample_dst_stage_id = 2;
        } 
        else{
            Stage prefill(Stage::StageType::TextFill);
            seq->stages.push_back(prefill);
            prefill.token_ids = token_ids;
            for(int i = 0;i < token_ids.size();i ++)prefill.position_ids.push_back(i);
            for(int i = 0;i < token_ids.size();i ++)prefill.cache_ids.push_back(i);
            prefill.sample = true;
            prefill.sample_dst_stage_id = 1;
        }
        int n_promp_tokens = token_ids.size();
        int max_tokens = sp.max_tokens != -1 ? sp.max_tokens : config.stage_config.default_max_tokens;
        for(int i = 0;i < max_tokens - 1;i ++){
            Stage decode(Stage::StageType::TextFill);
            seq->stages.push_back(decode);
            decode.position_ids.push_back(n_promp_tokens + i);
            decode.cache_ids.push_back(n_promp_tokens + i);
            decode.sample = true;
            decode.sample_dst_stage_id = seq->stages.size();
        }
        Stage empty(Stage::StageType::Empty);
        seq->stages.push_back(empty);

        // 3. schedule
        scheduler->schedule_new(seq);
        promise.set_value(true);
    });

    return future;
}

std::future<bool> Engine::add_request_async(std::string prompt,
    torch::Tensor pixel_value, 
    const SamplingParams& sp, 
    bool stream,
    OutputCallback callback){

    return add_request(
        std::move(prompt), 
        std::move(pixel_value),
        std::move(sp), 
        stream, 
        [callback = std::move(callback)](const RequestOutput& output){
            return callback(output);
        }
    );
}

BatchFuture Engine::add_requests_async(
    std::vector<std::string> prompts,
    std::vector<torch::Tensor> pixel_values, 
    std::vector<SamplingParams> sps, 
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
            std::move(sps[i]),
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
    if(this_step.size() == 0)return;
    // 2. batch sequence and execute
    std::vector<Sequence*> fill;
    std::vector<Sequence*> image_embed;
    for(int i = 0;i < this_step.size();i ++){
        Sequence* seq = this_step[i];
        Stage* stage = &seq->stages[seq->curr_stage_id];
        switch (stage->type)
        {
        case Stage::StageType::Empty:
            break;
        case Stage::StageType::TextFill:
            fill.push_back(seq);
            break;
        case Stage::StageType::ImageFill:
            fill.push_back(seq);
            break;
        case Stage::StageType::ImageEmbedFill:
            fill.push_back(seq);
            break;
        case Stage::StageType::ImageEmbed:
            image_embed.push_back(seq);
            break;
        default:
            CHECK(false) << "inavlid stage";
        }
    }
    execute_batch_fill(fill);
    if(config.batch_image_embed_forward)execute_batch_image_embed(fill);
    else CHECK(false) << "not implemented for now";
    // 3. schedule sequence
    for(int i = 0;i < this_step.size();i ++){
        Sequence* seq = this_step[i];
        seq->curr_stage_id ++;
        if(seq->curr_stage_id >= seq->stages.size()){
            printf("todo finish sequence");
        }else{
            scheduler->schedule_running(seq);
        }
    }
}


void Engine::execute_batch_fill(std::vector<Sequence*>& seqs){
    if(seqs.size() == 0)return;
    // 1. prepare input
    int num_sequences = seqs.size();
    std::vector<torch::Tensor>pixel_values;
    std::vector<torch::Tensor>image_features;
    bool has_image_fill = false;
    bool has_image_embed_fill = false;
    for(Sequence* seq : seqs){
        Stage* stage = &(seq->stages[seq->curr_stage_id]);
        if(stage->type == Stage::StageType::ImageFill){
            pixel_values.push_back(stage->pixel_values);
            has_image_fill = true;
        }
        else if(stage->type == Stage::StageType::ImageFill){
            image_features.push_back(stage->image_features);
            has_image_embed_fill = true;
        }
    }
    CHECK(!(has_image_fill && has_image_embed_fill)) << "not support pixel value and image embed batch";
    std::vector<int> token_ids;
    std::vector<int> position_ids;
    std::vector<int> q_seq_lens;
    std::vector<int> q_cu_seq_lens{0};
    int q_max_seq_len = 0;
    std::vector<int> selected_token_ids;
    bool all_sequences_decode;
    for(Sequence* seq : seqs){
        Stage* stage = &(seq->stages[seq->curr_stage_id]);
        for(int token_id : stage->token_ids)token_ids.push_back(token_id);
        for(int position_id : stage->position_ids)position_ids.push_back(position_id);
        q_seq_lens.push_back(stage->token_ids.size());
        if(stage->sample)selected_token_ids.push_back(token_ids.size() - 1);
        q_max_seq_len = std::max(q_max_seq_len, static_cast<int>(stage->token_ids.size()));
        q_cu_seq_lens.push_back(q_cu_seq_lens.back() + stage->token_ids.size());
    }
    all_sequences_decode = token_ids.size() == num_sequences;

    std::vector<int> layer_kv_seq_lens;
    std::vector<int> layer_paged_kv_last_page_len;
    std::vector<int> layer_kv_cu_seq_lens{0};
    std::vector<int> layer_block_tables;
    std::vector<int> layer_blocks_lens;
    std::vector<int> layer_cu_blocks_lens{0};
    std::vector<int> layer_new_cache_slots;

    for(Sequence* seq : seqs){
        Stage* stage = &(seq->stages[seq->curr_stage_id]);
        // 1. allocate memory if necessary
        int max_cache_id = 0;
        for(int cache_id : stage->cache_ids) max_cache_id = std::max(max_cache_id, cache_id);
        int n_need_tokens = max_cache_id + 1;
        seq->n_kv_cache_tokens = std::max(seq->n_kv_cache_tokens, n_need_tokens);
        int n_need_blocks = (n_need_tokens + config.memory_config.block_size - 1) / config.memory_config.block_size;
        if(seq->block_tables.size() < n_need_blocks){
            std::vector<int> blocks = kv_caches[0]->allocate(n_need_blocks - seq->block_tables.size());
            for(int block : blocks)seq->block_tables.push_back(block);
        }
        // 2. v2p
        for(int cache_id : stage->cache_ids){
            int block_id = cache_id / config.memory_config.block_size;
            int block_offset = cache_id % config.memory_config.block_size;
            int slot_id = seq->block_tables[block_id] * config.memory_config.block_size + block_offset;
            layer_new_cache_slots.push_back(slot_id);
        }
        layer_kv_seq_lens.push_back(seq->n_kv_cache_tokens);
        layer_kv_cu_seq_lens.push_back(layer_kv_cu_seq_lens.back() + seq->n_kv_cache_tokens);
        for(int block_id : seq->block_tables)layer_block_tables.push_back(block_id);
        layer_blocks_lens.push_back(seq->block_tables.size());
        layer_cu_blocks_lens.push_back(layer_cu_blocks_lens.back() + seq->block_tables.size());
        layer_paged_kv_last_page_len.push_back((seq->n_kv_cache_tokens + config.memory_config.block_size - 1) % config.memory_config.block_size + 1);
    }
    // 2. prepare tensor
    torch::TensorOptions int_options = torch::dtype(torch::kInt32).device(torch::kCUDA);
    auto ten_input_ids = torch::tensor(token_ids, int_options);
    auto ten_position_ids = torch::tensor(position_ids, int_options);
    auto ten_q_cu_seq_lens = torch::tensor(q_cu_seq_lens, int_options);
    auto ten_layer_new_cache_slots = torch::tensor(layer_new_cache_slots, int_options);
    auto ten_layer_block_tables = torch::tensor(layer_block_tables, int_options);
    auto ten_layer_cu_blocks_lens = torch::tensor(layer_cu_blocks_lens, int_options);
    auto ten_layer_kv_cu_seq_lens = torch::tensor(layer_kv_cu_seq_lens, int_options);
    auto ten_layer_paged_kv_last_page_len = torch::tensor(layer_paged_kv_last_page_len, int_options);

    ModelParameters model_params;
    if(config.memory_config.memory_management_policy == MemoryConfig::MemoryManagementPolicy::VANILLA){
        for(int i = 0;i < model_config.n_layers;i ++){
            AttentionParameters attention_params;
            attention_params.kv_cache = kv_caches[i].get();
            attention_params.q_cu_seq_lens = ten_q_cu_seq_lens;
            attention_params.k_cu_seq_lens = ten_layer_kv_cu_seq_lens;
            attention_params.paged_kv_last_page_len = ten_layer_paged_kv_last_page_len;
            attention_params.new_cache_slots = ten_layer_new_cache_slots;
            attention_params.block_tables = ten_layer_block_tables;
            attention_params.cu_block_lens = ten_layer_cu_blocks_lens;
            attention_params.num_sequences = num_sequences;
            attention_params.all_sequences_decode = all_sequences_decode;
            attention_params.q_max_seq_len = 128;
            attention_params.k_max_seq_len = 128;
            model_params.attention_params.push_back(attention_params);
        }
    }else{
        CHECK(false) << "not implemented shared memory management policy";
    }
    torch::Tensor ten_pixel_values;
    if (pixel_values.size())ten_pixel_values = torch::cat(pixel_values, 0).to(options_);
    torch::Tensor ten_image_featues;
    if (image_features.size())ten_image_featues = torch::cat(image_features, /*dim*/0).to(options_);
    // 3. prepare forward sample
    auto logits = model->forward(ten_input_ids, ten_pixel_values, ten_image_featues, ten_position_ids, model_params);

    if(selected_token_ids.size()){
        auto ten_selected_token_ids = torch::tensor(selected_token_ids, int_options);
        auto sample_token_ids = torch::argmax(logits.index_select(0, ten_selected_token_ids), -1, false);
        int i = 0;
        for(Sequence* seq : seqs){
            Stage* stage = &(seq->stages[seq->curr_stage_id]);
            if(stage->sample){
                int next_token_id = sample_token_ids[i].item<int>();
                seq->stages[stage->sample_dst_stage_id].token_ids.push_back(next_token_id);
                seq->request_output.output_token_ids.push_back(next_token_id);
                i ++;
            }
        }
    }
}

void Engine::execute_batch_image_embed(std::vector<Sequence*>& seqs){
    if(seqs.size() == 0)return;
    std::vector<int> n_images;
    std::vector<torch::Tensor>batch_pixel_values;
    for(Sequence* seq : seqs){
        Stage* stage = &(seq->stages[seq->curr_stage_id]);
        auto pixel_values = stage->pixel_values.to(options_);
        batch_pixel_values.push_back(pixel_values);
        n_images.push_back(pixel_values.size(0));
    }
    auto pixel_values = torch::cat(batch_pixel_values, /*dim*/0);

    ModelParameters model_params;
    auto image_features = model->image_embed(pixel_values, model_params);

    int left = 0;
    for(int i = 0;i < seqs.size();i ++){
        Sequence* seq = seqs[i];
        Stage* stage = &(seq->stages[seq->curr_stage_id]);
        int right = left + n_images[i];
        seq->stages[stage->image_feature_dst_stage_id].image_features = image_features.index({torch::indexing::Slice(left, right), torch::indexing::Slice(), torch::indexing::Slice()});
        left += n_images[i];
    }
}

}