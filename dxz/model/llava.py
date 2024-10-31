import torch
from torch import nn, Tensor
from transformers import LlavaConfig, LlamaConfig
from dxz.model.parameters import InputParameters
from dxz.memory.kv_cache import KVCache
from dxz.model.llama import LlamaDecoderLayer, LlamaRMSNorm
from dxz.model.clip import CLIPVisionModel
from typing import Optional
import math

class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config)
    
    def forward(self, input_embeds: Tensor, position_ids: Tensor, kv_caches: list[KVCache], input_params: InputParameters) -> Tensor:
        hidden_states = input_embeds
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, position_ids, kv_caches[i], input_params)
        return self.norm(hidden_states)

class LlamaForCausalLM(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def forward(self, input_embeds: Tensor, position_ids: Tensor, kv_caches: list[KVCache], input_params: InputParameters) -> Tensor:
        hidden_state = self.model(input_embeds, position_ids, kv_caches, input_params)
        logits = self.lm_head(hidden_state)
        return logits

class LlavaMultiModalProjector(nn.Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()
        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.act = nn.GELU()
    
    def forward(self, image_features: Tensor) -> Tensor:
        # image_features (n_images, n_tokens, vision_hidden_size)
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states

class LlavaForConditionalGeneration(nn.Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()
        self.config = config
        # assert config.vision_feature_select_strategy == 'default'
        self.vision_tower = CLIPVisionModel(config.vision_config)
        self.multi_modal_projector = LlavaMultiModalProjector(config)
        self.language_model = LlamaForCausalLM(config.text_config)
    
    def forward(self, input_ids: Tensor, pixel_values: Optional[Tensor], position_ids: Tensor, kv_caches: Tensor, input_params: InputParameters) -> Tensor:
        # 1. compute input embeds
        input_embeds = self.language_model.model.embed_tokens(input_ids) # (n_tokens, n_embeds)

        if pixel_values is not None:
            # 2. compute image embeds
            _, _, all_hidden_states = self.vision_tower(pixel_values)
            hidden_states = all_hidden_states[self.config.vision_feature_layer] # (n_images, 577, 1024)
            selected_image_feature = hidden_states[:, 1:] # (n_images, 576, 1024)
            image_features = self.multi_modal_projector(selected_image_feature)

            # 3. merge embeds
            image_token_mask = input_ids == self.config.image_token_index 
            new_token_position = torch.cumsum(image_token_mask * (576 - 1) + 1, dim=-1) - 1

            n_tokens = input_ids.shape[0] + image_features.shape[0] * (image_features.shape[1] - 1)

            text_overwrite_idx = new_token_position[~image_token_mask]

            merged_embed = torch.zeros(size=(n_tokens, self.config.text_config.hidden_size), dtype=input_embeds.dtype)
            merged_embed[text_overwrite_idx, :] = input_embeds[~image_token_mask, :]

            image_overwrite_mask = torch.ones(size=(n_tokens, ), dtype=torch.bool)
            image_overwrite_mask[text_overwrite_idx] = False
            merged_embed[image_overwrite_mask, :] = image_features.reshape(-1, self.config.text_config.hidden_size)

        if pixel_values is not None:
            position_ids = torch.arange(n_tokens, dtype=torch.int, device=input_ids.device) # todo check prefill or decode

        # 4. compute logits
        if pixel_values is not None:
            logits = self.language_model(merged_embed, position_ids, kv_caches, input_params)
        else:
            logits = self.language_model(input_embeds, position_ids, kv_caches, input_params)

        return logits
