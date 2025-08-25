import torch
from torch import nn, Tensor
from hydrainfer.model_parallel.process_group import ProcessGroup
from dataclasses import dataclass

@dataclass
class EPMoEConfig:
    num_experts: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    rnormalize: bool
    use_grouped_topk: bool
    num_expert_group: int
    topk_group: int
    correction_bias: Tensor
    process_group: ProcessGroup

@dataclass
class QuantConfig:
    quant_method: str
    weight_block_size: list[int]
    activation_dynamic: bool

class EPMoE(nn.Module):
    def __init__(self, config: EPMoEConfig, quant_config: QuantConfig):
        super().__init__()
        assert config.num_experts % config.process_group.world_size == 0
        num_experts_per_partition = config.num_experts // config.process_group.world_size
        ep_rank = config.process_group.rank
        start_expert_id =  ep_rank * num_experts_per_partition
        end_expert_id = start_expert_id + num_experts_per_partition - 1;
        weight_dtype = torch.half
        scale_dtype = torch.float32
        if quant_config.quant_method == 'fp8':
            weight_dtype = torch.float8_e4m3fn
            use_fp8_w8a8 = True
            weight_block_size: list[int] = quant_config.weight_block_size
            block_quant: bool = len(quant_config.weight_block_size) > 0
            activation_dynamic: bool = quant_config.activation_dynamic

         


# class EPMoEImpl : public torch::nn::Module {
#  public:
#     // Fused gate_up_proj (column parallel)
#     w13_weight_ = register_parameter(
#         "w13_weight",
#         torch::empty(
#             {num_experts_per_partition_, intermediate_size_ * 2, hidden_size_},
#             weight_options),
#         /*requires_grad=*/false);
#     // down_proj (row parallel)
#     w2_weight_ = register_parameter(
#         "w2_weight",
#         torch::empty(
#             {num_experts_per_partition_, hidden_size_, intermediate_size_},
#             weight_options),
#         /*requires_grad=*/false);
#     // weight scale
#     if (block_quant_) {
#       const int32_t block_n = weight_block_size_[0];
#       const int32_t block_k = weight_block_size_[1];
#       w13_weight_scale_ = register_parameter(
#           "w13_weight_scale",
#           torch::ones({num_experts_per_partition_,
#                        2 * ((intermediate_size_ + block_n - 1) / block_n),
#                        (hidden_size_ + block_k - 1) / block_k},
#                       scale_options),
#           /*requires_grad=*/false);
#       w2_weight_scale_ = register_parameter(
#           "w2_weight_scale",
#           torch::ones({num_experts_per_partition_,
#                        (hidden_size_ + block_n - 1) / block_n,
#                        (intermediate_size_ + block_k - 1) / block_k},
#                       scale_options),
#           /*requires_grad=*/false);
#     } else {
#       w13_weight_scale_ = register_parameter(
#           "w13_weight_scale",
#           torch::ones({num_experts_per_partition_}, scale_options),
#           /*requires_grad=*/false);
#       w2_weight_scale_ = register_parameter(
#           "w2_weight_scale",
#           torch::ones({num_experts_per_partition_}, scale_options),
#           /*requires_grad=*/false);
#     }
#     // input scale
#     w13_input_scale_ = register_parameter(
#         "w13_input_scale",
#         torch::ones({num_experts_per_partition_}, scale_options),
#         /*requires_grad=*/false);
#     w2_input_scale_ = register_parameter(
#         "w2_input_scale",
#         torch::ones({num_experts_per_partition_}, scale_options),
#         /*requires_grad=*/false);

#     ep_moe_triton_kernel_ = std::make_unique<EPMoETritonKernel>(options);
#     quant_fp8_triton_kernel_ = std::make_unique<QuantFP8TritonKernel>(options);
#   }

#   torch::Tensor forward(torch::Tensor hidden_states,
#                         torch::Tensor router_logits) {
#     torch::Tensor topk_weights, topk_ids;
#     std::tie(topk_weights, topk_ids) = select_experts(hidden_states,
#                                                       router_logits,
#                                                       top_k_,
#                                                       use_grouped_topk_,
#                                                       renormalize_,
#                                                       topk_group_,
#                                                       num_expert_group_,
#                                                       correction_bias_);
#     // topk_ids = topk_ids.to(torch::kInt64);
#     torch::TensorOptions INT_OPTIONS =
#         hidden_states.options().dtype(torch::kInt64);
#     torch::TensorOptions FLOAT_OPTIONS = hidden_states.options();
#     torch::TensorOptions GEMM_OPTIONS = hidden_states.options();
#     if (use_fp8_w8a8_ && !block_quant_) {
#       GEMM_OPTIONS = GEMM_OPTIONS.dtype(torch::kFloat8_e4m3fn);
#       if (activation_dynamic_) {
#         auto max_value =
#             torch::max(hidden_states)
#                 .repeat({num_experts_per_partition_})
#                 .to(hidden_states.options().dtype(torch::kFloat32));
#         w13_input_scale_ = max_value / 448.0f;  // fp8 max
#       }
#     }

#     torch::Tensor seg_indptr = torch::zeros({num_experts_ + 1}, INT_OPTIONS);
#     torch::Tensor src2dst = torch::empty({topk_ids.numel()}, INT_OPTIONS);
#     torch::Tensor reorder_topk_ids = kernel::run_moe_ep_preproess(
#         topk_ids, num_experts_, src2dst, seg_indptr);

#     // PreReorder
#     torch::Tensor gateup_input = torch::empty(
#         {hidden_states.size(0) * top_k_, hidden_states.size(1)}, GEMM_OPTIONS);
#     ep_moe_triton_kernel_->pre_reorder_triton(hidden_states,
#                                               gateup_input,
#                                               src2dst,
#                                               topk_ids,
#                                               w13_input_scale_,
#                                               start_expert_id_,
#                                               end_expert_id_,
#                                               top_k_,
#                                               hidden_states.size(1));

#     torch::Tensor seg_indptr_cur_rank =
#         seg_indptr.index({ISlice(start_expert_id_, end_expert_id_ + 2)});
#     torch::Tensor weight_indices_cur_rank =
#         torch::arange(num_experts_per_partition_, INT_OPTIONS);

#     std::vector<int64_t> weight_block_size{128, 128};
#     // GroupGemm-0
#     torch::Tensor gateup_output = torch::empty(
#         {gateup_input.size(0), w13_weight_.size(1)}, FLOAT_OPTIONS);
#     torch::Tensor input_a, a_scale;
#     if (block_quant_) {
#       int32_t block_n = weight_block_size_[0];
#       int32_t block_k = weight_block_size_[1];
#       std::tie(input_a, a_scale) =
#           quant_fp8_triton_kernel_->per_token_group_quant_fp8(
#               gateup_input, weight_block_size_[1]);
#       CHECK((input_a.size(-1) + block_k - 1) / block_k == a_scale.size(-1));
#       CHECK((w13_weight_.size(-2) + block_n - 1) / block_n ==
#             w13_weight_scale_.size(-2));
#       CHECK((w13_weight_.size(-1) + block_k - 1) / block_k ==
#             w13_weight_scale_.size(-1));
#     } else {
#       input_a = gateup_input;
#       a_scale = w13_input_scale_;
#     }
#     ep_moe_triton_kernel_->grouped_gemm_triton(input_a,
#                                                w13_weight_,
#                                                gateup_output,
#                                                num_experts_per_partition_,
#                                                /*weight_column_major*/ true,
#                                                seg_indptr_cur_rank,
#                                                weight_indices_cur_rank,
#                                                a_scale,
#                                                w13_weight_scale_,
#                                                use_fp8_w8a8_,
#                                                weight_block_size);
#     gateup_input.reset();
#     // Act
#     torch::Tensor down_input = torch::empty(
#         {gateup_output.size(0), gateup_output.size(1) / 2}, GEMM_OPTIONS);
#     ep_moe_triton_kernel_->silu_and_mul_triton(gateup_output,
#                                                down_input,
#                                                gateup_output.size(1),
#                                                reorder_topk_ids,
#                                                w2_input_scale_,
#                                                start_expert_id_,
#                                                end_expert_id_);
#     gateup_output.reset();
#     // GroupGemm-1
#     torch::Tensor down_output =
#         torch::empty({down_input.size(0), w2_weight_.size(1)}, FLOAT_OPTIONS);
#     if (block_quant_) {
#       int32_t block_n = weight_block_size_[0];
#       int32_t block_k = weight_block_size_[1];
#       std::tie(input_a, a_scale) =
#           quant_fp8_triton_kernel_->per_token_group_quant_fp8(
#               down_input, weight_block_size_[1]);
#       CHECK((input_a.size(-1) + block_k - 1) / block_k == a_scale.size(-1));
#       CHECK((w2_weight_.size(-2) + block_n - 1) / block_n ==
#             w2_weight_scale_.size(-2));
#       CHECK((w2_weight_.size(-1) + block_k - 1) / block_k ==
#             w2_weight_scale_.size(-1));
#     } else {
#       input_a = down_input;
#       a_scale = w2_input_scale_;
#     }
#     ep_moe_triton_kernel_->grouped_gemm_triton(input_a,
#                                                w2_weight_,
#                                                down_output,
#                                                num_experts_per_partition_,
#                                                /*weight_column_major*/ true,
#                                                seg_indptr_cur_rank,
#                                                weight_indices_cur_rank,
#                                                a_scale,
#                                                w2_weight_scale_,
#                                                use_fp8_w8a8_,
#                                                weight_block_size);
#     down_input.reset();
#     // PostReorder
#     torch::Tensor output = torch::empty_like(hidden_states);
#     ep_moe_triton_kernel_->post_reorder_triton(down_output,
#                                                output,
#                                                src2dst,
#                                                topk_ids,
#                                                topk_weights,
#                                                start_expert_id_,
#                                                end_expert_id_,
#                                                top_k_,
#                                                hidden_states.size(1));
#     down_output.reset();
#     return output;
#   }

#   // load the weight from the checkpoint
#   void load_state_dict(const StateDict& state_dict) {
#     // TODO(liangzhiwei20): If checkpoint is fp16, quantize in place.
#     // But we need to check whether the checkpoint is fp16 or fp32.
#     load_state_dict_w13(state_dict);
#     load_state_dict_w2(state_dict);
#     load_state_dict_w13_w_s(state_dict);
#     load_state_dict_w2_w_s(state_dict);
#     load_state_dict_w13_i_s(state_dict);
#     load_state_dict_w2_i_s(state_dict);
#     // TODO(liangzhiwei20): support fp8 tensor quantization
#     // process_weights_after_loading_w13();
#   }

#   void load_state_dict_w13(const StateDict& state_dict) {
#     // return if the weight is already loaded
#     if (w13_weight_is_loaded_) {
#       return;
#     }

#     if (w13_weight_list_.size() < num_experts_per_partition_ * 2) {
#       w13_weight_list_.resize(num_experts_per_partition_ * 2);
#     }
#     for (int e_id = start_expert_id_; e_id <= end_expert_id_; e_id++) {
#       int index = e_id % num_experts_per_partition_;
#       const std::string w1_tensor_name =
#           std::to_string(e_id) + ".gate_proj." + "weight";
#       torch::Tensor w1_tensor = state_dict.get_tensor(w1_tensor_name);
#       if (w1_tensor.defined()) {
#         w13_weight_.index_put_({ISlice(index, index + 1),
#                                 ISlice(0, intermediate_size_),
#                                 torch::indexing::Ellipsis},
#                                w1_tensor.unsqueeze(0));
#         w13_weight_list_[index * 2] = torch::ones({1});
#       }
#       const std::string w3_tensor_name =
#           std::to_string(e_id) + ".up_proj." + "weight";
#       torch::Tensor w3_tensor = state_dict.get_tensor(w3_tensor_name);
#       if (w3_tensor.defined()) {
#         w13_weight_.index_put_({ISlice(index, index + 1),
#                                 ISlice(intermediate_size_),
#                                 torch::indexing::Ellipsis},
#                                w3_tensor.unsqueeze(0));
#         w13_weight_list_[index * 2 + 1] = torch::ones({1});
#       }
#     }
#     const bool all_loaded =
#         std::all_of(w13_weight_list_.begin(),
#                     w13_weight_list_.end(),
#                     [](const torch::Tensor& t) { return t.defined(); });
#     if (all_loaded) {
#       w13_weight_is_loaded_ = true;
#     }
#   }

#   void load_state_dict_w2(const StateDict& state_dict) {
#     // return if the weight is already loaded
#     if (w2_weight_is_loaded_) {
#       return;
#     }

#     if (w2_weight_list_.size() < num_experts_per_partition_) {
#       w2_weight_list_.resize(num_experts_per_partition_);
#     }

#     // std::vector<torch::Tensor> w13_tensors;
#     for (int e_id = start_expert_id_; e_id <= end_expert_id_; e_id++) {
#       int index = e_id % num_experts_per_partition_;
#       const std::string w2_tensor_name =
#           std::to_string(e_id) + ".down_proj." + "weight";
#       torch::Tensor w2_tensor = state_dict.get_tensor(w2_tensor_name);
#       if (w2_tensor.defined()) {
#         w2_weight_.index_put_(
#             {ISlice(index, index + 1), torch::indexing::Ellipsis},
#             w2_tensor.unsqueeze(0));
#         w2_weight_list_[index] = torch::ones({1});
#       }
#     }
#     const bool all_loaded =
#         std::all_of(w2_weight_list_.begin(),
#                     w2_weight_list_.end(),
#                     [](const torch::Tensor& t) { return t.defined(); });
#     if (all_loaded) {
#       w2_weight_is_loaded_ = true;
#     }
#   }

#   void load_state_dict_w13_w_s(const StateDict& state_dict) {
#     // return if the weight is already loaded
#     if (w13_weight_scale_is_loaded_) {
#       return;
#     }

#     if (!use_fp8_w8a8_) {
#       w13_weight_scale_is_loaded_ = true;
#       return;
#     }

#     if (w13_weight_scale_list_.size() < num_experts_per_partition_ * 2) {
#       w13_weight_scale_list_.resize(num_experts_per_partition_ * 2);
#     }

#     const int32_t block_n = weight_block_size_[0];
#     const int32_t size_n = (intermediate_size_ + block_n - 1) / block_n;
#     for (int e_id = start_expert_id_; e_id <= end_expert_id_; e_id++) {
#       int index = e_id % num_experts_per_partition_;
#       std::string tensor_name;
#       if (block_quant_) {
#         tensor_name = "weight_scale_inv";
#       } else {
#         tensor_name = "weight_scale";
#       }
#       const std::string w1_tensor_name =
#           std::to_string(e_id) + ".gate_proj." + tensor_name;
#       torch::Tensor w1_tensor = state_dict.get_tensor(w1_tensor_name);
#       if (w1_tensor.defined()) {
#         w13_weight_scale_.index_put_({ISlice(index, index + 1),
#                                       ISlice(0, size_n),
#                                       torch::indexing::Ellipsis},
#                                      w1_tensor.unsqueeze(0));
#         w13_weight_scale_list_[index * 2] = torch::ones({1});
#       }
#       const std::string w3_tensor_name =
#           std::to_string(e_id) + ".up_proj." + tensor_name;
#       torch::Tensor w3_tensor = state_dict.get_tensor(w3_tensor_name);
#       if (w3_tensor.defined()) {
#         w13_weight_scale_.index_put_({ISlice(index, index + 1),
#                                       ISlice(size_n),
#                                       torch::indexing::Ellipsis},
#                                      w3_tensor.unsqueeze(0));
#         w13_weight_scale_list_[index * 2 + 1] = torch::ones({1});
#       }
#     }
#     const bool all_loaded =
#         std::all_of(w13_weight_scale_list_.begin(),
#                     w13_weight_scale_list_.end(),
#                     [](const torch::Tensor& t) { return t.defined(); });
#     if (all_loaded) {
#       w13_weight_scale_is_loaded_ = true;
#     }
#   }

#   void load_state_dict_w2_w_s(const StateDict& state_dict) {
#     // return if the weight is already loaded
#     if (w2_weight_scale_is_loaded_) {
#       return;
#     }

#     if (!use_fp8_w8a8_) {
#       w2_weight_scale_is_loaded_ = true;
#       return;
#     }

#     if (w2_weight_scale_list_.size() < num_experts_per_partition_) {
#       w2_weight_scale_list_.resize(num_experts_per_partition_);
#     }

#     for (int e_id = start_expert_id_; e_id <= end_expert_id_; e_id++) {
#       int index = e_id % num_experts_per_partition_;
#       std::string tensor_name;
#       if (block_quant_) {
#         tensor_name = "weight_scale_inv";
#       } else {
#         tensor_name = "weight_scale";
#       }
#       const std::string w2_tensor_name =
#           std::to_string(e_id) + ".down_proj." + tensor_name;
#       torch::Tensor w2_tensor = state_dict.get_tensor(w2_tensor_name);
#       if (w2_tensor.defined() && !w2_weight_scale_list_[index].defined()) {
#         w2_weight_scale_.index_put_(
#             {ISlice(index, index + 1), torch::indexing::Ellipsis},
#             w2_tensor.unsqueeze(0));
#         w2_weight_scale_list_[index] = torch::ones({1});
#       }
#     }
#     const bool all_loaded =
#         std::all_of(w2_weight_scale_list_.begin(),
#                     w2_weight_scale_list_.end(),
#                     [](const torch::Tensor& t) { return t.defined(); });
#     if (all_loaded) {
#       w2_weight_scale_is_loaded_ = true;
#     }
#   }

#   void load_state_dict_w13_i_s(const StateDict& state_dict) {
#     // return if the weight is already loaded
#     if (w13_input_scale_is_loaded_) {
#       return;
#     }

#     if (!use_fp8_w8a8_ || activation_dynamic_) {
#       w13_input_scale_is_loaded_ = true;
#       return;
#     }

#     if (w13_input_scale_list_.size() < num_experts_per_partition_ * 2) {
#       w13_input_scale_list_.resize(num_experts_per_partition_ * 2);
#     }

#     for (int e_id = start_expert_id_; e_id <= end_expert_id_; e_id++) {
#       int index = e_id % num_experts_per_partition_;
#       const std::string w1_tensor_name =
#           std::to_string(e_id) + ".gate_proj.input_scale";
#       torch::Tensor w1_tensor = state_dict.get_tensor(w1_tensor_name);
#       if (w1_tensor.defined() && !w13_input_scale_list_[index * 2].defined()) {
#         w13_input_scale_list_[index * 2] = w1_tensor.clone();
#       }
#       const std::string w3_tensor_name =
#           std::to_string(e_id) + ".up_proj.input_scale";
#       torch::Tensor w3_tensor = state_dict.get_tensor(w3_tensor_name);
#       if (w3_tensor.defined() &&
#           !w13_input_scale_list_[index * 2 + 1].defined()) {
#         w13_input_scale_list_[index * 2 + 1] = w3_tensor.clone();
#       }
#     }
#     const bool all_loaded =
#         std::all_of(w13_input_scale_list_.begin(),
#                     w13_input_scale_list_.end(),
#                     [](const torch::Tensor& t) { return t.defined(); });
#     if (all_loaded) {
#       std::vector<torch::Tensor> w13_tensors(num_experts_per_partition_);
#       for (int e_id = 0; e_id < num_experts_per_partition_; e_id++) {
#         torch::Tensor w13_tensor =
#             torch::cat({w13_input_scale_list_[e_id * 2],
#                         w13_input_scale_list_[e_id * 2 + 1]},
#                        /*dim*/ 0);
#         w13_tensors[e_id] = w13_tensor.max();
#       }
#       w13_input_scale_list_.clear();
#       const torch::Tensor merged_weight = torch::stack(w13_tensors, /*dim=*/0);

#       w13_input_scale_.copy_(merged_weight);
#       w13_input_scale_is_loaded_ = true;
#       w13_tensors.clear();
#     }
#   }

#   void load_state_dict_w2_i_s(const StateDict& state_dict) {
#     // return if the weight is already loaded
#     if (w2_input_scale_is_loaded_) {
#       return;
#     }

#     if (!use_fp8_w8a8_ || activation_dynamic_) {
#       w2_input_scale_is_loaded_ = true;
#       return;
#     }

#     if (w2_input_scale_list_.size() < num_experts_per_partition_) {
#       w2_input_scale_list_.resize(num_experts_per_partition_);
#     }

#     // std::vector<torch::Tensor> w13_tensors;
#     for (int e_id = start_expert_id_; e_id <= end_expert_id_; e_id++) {
#       int index = e_id % num_experts_per_partition_;
#       const std::string w2_tensor_name =
#           std::to_string(e_id) + ".down_proj.input_scale";
#       torch::Tensor w2_tensor = state_dict.get_tensor(w2_tensor_name);
#       if (w2_tensor.defined() && !w2_input_scale_list_[index].defined()) {
#         w2_input_scale_list_[index] = w2_tensor.max();
#       }
#     }
#     const bool all_loaded =
#         std::all_of(w2_input_scale_list_.begin(),
#                     w2_input_scale_list_.end(),
#                     [](const torch::Tensor& t) { return t.defined(); });
#     if (all_loaded) {
#       const torch::Tensor merged_weight =
#           torch::stack(w2_input_scale_list_, /*dim=*/0);
#       w2_input_scale_.copy_(merged_weight);
#       w2_input_scale_is_loaded_ = true;
#       w2_input_scale_list_.clear();
#     }
#   }

#   // whether the weight is loaded
#   void verify_loaded_weights(const std::string& prefix = "") const {
#     CHECK(w13_weight_is_loaded_)
#         << "weight is not loaded for " << prefix + "w13_weight";
#     CHECK(w2_weight_is_loaded_)
#         << "bias is not loaded for " << prefix + "w2_weight";
#     CHECK(w13_input_scale_is_loaded_)
#         << "weight is not loaded for " << prefix + "w13_input_scale";
#     CHECK(w2_input_scale_is_loaded_)
#         << "bias is not loaded for " << prefix + "w2_input_scale";
#     CHECK(w13_weight_scale_is_loaded_)
#         << "weight is not loaded for " << prefix + "w13_weight_scale";
#     CHECK(w2_weight_scale_is_loaded_)
#         << "bias is not loaded for " << prefix + "w2_weight_scale";
#   }

#  private:
#   int64_t top_k_;
#   int64_t num_experts_;
#   int64_t num_experts_per_partition_;
#   int64_t start_expert_id_;
#   int64_t end_expert_id_;
#   int64_t hidden_size_;
#   int64_t intermediate_size_;
#   bool renormalize_;
#   bool use_grouped_topk_;
#   int64_t num_expert_group_;
#   int64_t topk_group_;
#   torch::Tensor correction_bias_;

#   DEFINE_FUSED_WEIGHT(w13_weight);
#   DEFINE_FUSED_WEIGHT(w2_weight);
#   DEFINE_FUSED_WEIGHT(w13_weight_scale);
#   DEFINE_FUSED_WEIGHT(w2_weight_scale);
#   DEFINE_FUSED_WEIGHT(w13_input_scale);
#   DEFINE_FUSED_WEIGHT(w2_input_scale);

#   // whether the weight be serialized in fp8 format
#   bool use_fp8_w8a8_ = false;

#   // whether block quant
#   bool block_quant_ = false;

#   // weight block size
#   std::vector<int64_t> weight_block_size_ = {};

#   // whether activation scheme is dynamic
#   bool activation_dynamic_ = false;

#   std::unique_ptr<EPMoETritonKernel> ep_moe_triton_kernel_;
#   std::unique_ptr<QuantFP8TritonKernel> quant_fp8_triton_kernel_;
# };
# TORCH_MODULE(EPMoE);