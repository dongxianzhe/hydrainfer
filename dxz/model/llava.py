import os
import safetensors.torch
import torch
from torch import nn, Tensor
from transformers import LlavaConfig, LlamaConfig, CLIPVisionConfig
from dxz.model.parameters import ModelParameters
from dxz.model.llama import LlamaDecoderLayer, LlamaRMSNorm
from typing import Optional, Union
from dxz.model.clip import CLIPEncoderLayer, CLIPVisionEmbeddings

class CLIPEncoder(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.layers = nn.ModuleList([CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])
    
    def forward(self, hidden_states: Tensor, vision_feature_layer: int, model_params: ModelParameters) -> Tensor:
        for i, encoder_layer in enumerate(self.layers[:(vision_feature_layer + len(self.layers)) % len(self.layers) + 1]): #
            return_scores = i==(vision_feature_layer + len(self.layers)) % len(self.layers)
            if return_scores:
                hidden_states, scores = encoder_layer(hidden_states, return_scores=return_scores)
                model_params.clip_scores = scores
            else:
                hidden_states = encoder_layer(hidden_states, return_scores=return_scores)
            
        return hidden_states

class CLIPVisionTransformer(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(self, pixel_values: Tensor, vision_feature_layer: int, model_params: ModelParameters) -> Tensor:
        # pixel_values (batch_size, channels, width, height)
        hidden_states = self.embeddings(pixel_values) # (batch_size, n_tokens, hidden_size)
        hidden_states = self.pre_layrnorm(hidden_states)
        hidden_states = self.encoder(hidden_states, vision_feature_layer, model_params)
        return hidden_states

class CLIPVisionModel(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.vision_model = CLIPVisionTransformer(config)

    def forward(self, pixel_values: Tensor, vision_feature_layer: int, model_params: ModelParameters) -> Tensor:
        # pixel_values (n_pictures, n_channels, width, height)
        return self.vision_model(pixel_values, vision_feature_layer, model_params)


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, layer_id) for layer_id in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config)
    
    def forward(self, input_embeds: Tensor, position_ids: Tensor, model_params: ModelParameters) -> Tensor:
        hidden_states = input_embeds
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, position_ids, model_params)
        return self.norm(hidden_states)

class LlamaForCausalLM(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def forward(self, input_embeds: Tensor, position_ids: Tensor, model_params: ModelParameters) -> Tensor:
        hidden_state = self.model(input_embeds, position_ids, model_params)
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
    
    def forward(self, input_ids: Tensor, pixel_values: Optional[Tensor], image_features: Optional[Tensor], position_ids: Tensor, model_params: ModelParameters) -> Tensor:
        # input_ids    (n_tokens + m_tokens) n_tokens is text tokens, m_tokens is n_image * 576, n_image is number of image
        # pixel_values (n_images, n_channels, height, width)
        # position_ids (n_tokens + m_tokens)
        # image_features (m_tokens, hidden_size)

        # 1. compute input embeds
        input_embeds = self.input_embed(input_ids) # (n_tokens, n_embeds)

        if pixel_values is not None:
            image_overwrite_mask = input_ids == self.config.image_token_index
            # 2. compute image embeds
            image_features = self.image_embed(pixel_values, model_params)

            # 3. merge embeds
            embeds = self.merge_embed(input_embeds, image_features, image_overwrite_mask)

        if image_features is not None:
            image_overwrite_mask = input_ids == self.config.image_token_index
            embeds = self.merge_embed(input_embeds, image_features, image_overwrite_mask)
        # 4. compute logits
        logits = self.language_model(input_embeds, position_ids, model_params)

        return logits

    
    @classmethod
    def from_safetensor(cls, model_weights_path: str, dtype: torch.dtype, device: torch.device):
        # 1. create model
        config = LlavaConfig.from_pretrained(model_weights_path)
        torch.set_default_dtype(dtype)
        with torch.device(device):
            model = cls(config)
        torch.set_default_dtype(torch.float)

        # 2. load weights
        state_dict = model.state_dict()
        loaded_set = set()
        for entry in os.scandir(model_weights_path):
            if entry.is_file() and os.path.splitext(entry.name)[1] == '.safetensors':
                print(f'load safetensor from {entry.path}')
                for name, weight in safetensors.torch.load_file(entry.path).items():
                    state_dict[name].data.copy_(weight)
                    loaded_set.add(name)
        
        model.load_state_dict(state_dict)
        model.to(dtype)
        model.eval()

        # 3. verify
        assert len(loaded_set) == len(state_dict)

        return model

    def get_num_image_token_ids(self) -> int:
        return 576

    def input_embed(self, input_ids: Tensor) -> Tensor:
        input_embeds = self.language_model.model.embed_tokens(input_ids)
        return input_embeds

    def image_embed(self, pixel_values: Tensor, model_params: ModelParameters) -> Tensor:
        hidden_states = self.vision_tower(pixel_values, self.config.vision_feature_layer, model_params)
        selected_image_feature = hidden_states[:, 1:] # (n_images, 576, 1024)
        if model_params.embed_token_pruning_params and model_params.embed_token_pruning_params.get('policy', "") == 'focal':
            from dxz.layer import token_prunning
            selected_image_feature = token_prunning.focal_prunning(
                selected_image_feature,
                model_params.clip_scores[:, :, 1:, 1:], # (batch_size, n_images, 576, 576)
                n_output_tokens=model_params.embed_token_pruning_params['n_output_tokens'],
                strategy='rank',
            )
            
        image_features = self.multi_modal_projector(selected_image_feature)
        return image_features

    def merge_embed(self, input_embeds: Tensor, image_features: Tensor, image_overwrite_mask: Tensor) -> Tensor:
        # input_embeds   (n_tokens + m_tokens, hidden_size) n_tokens is text tokens, m_tokens is n_image * 576, n_image is number of image
        # image_features (n_images, 576, hidden_size) n_image * 576 should equal to m_tokens
        # image_over_write_idx = (n_tokens + m_tokens, )
        input_embeds[image_overwrite_mask, :] = image_features.view(-1, input_embeds.shape[-1])
        return input_embeds