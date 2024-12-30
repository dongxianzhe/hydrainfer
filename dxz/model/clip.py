import torch
from torch import nn, Tensor
from transformers import CLIPVisionConfig
import math
from dxz.model.parameters import VisionModelParameters, VisionModelOutput
from dxz.layer.activation import QuickGELUActivation

class CLIPSdpaAttention(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.n_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads 
        from dxz.layer.attention import TorchMultiHeadAttention
        self.torch_attention = TorchMultiHeadAttention(self.n_heads, self.head_dim)
        from dxz.layer.attention import FlashMultiHeadAttention
        self.flash_attention = FlashMultiHeadAttention(self.n_heads, self.head_dim)
    
    def forward(self, hidden_states: Tensor, return_scores: bool = False) -> Tensor:
        batch_size, n_tokens, embed_dim = hidden_states.size()
        query = self.q_proj(hidden_states)
        key   = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        if return_scores:
            o, scores = self.torch_attention(query, key, value, return_scores)
            o = self.out_proj(o)
            return o, scores
        else:
            o = self.flash_attention(query, key, value)
            o = self.out_proj(o)
            return o



class CLIPMLP(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.activation_fn = QuickGELUActivation()
    
    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class CLIPEncoderLayer(nn.Module):
    def __init__(self, config: CLIPVisionConfig, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.self_attn = CLIPSdpaAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(self, hidden_states: Tensor, vision_feature_layer: int, model_params: VisionModelParameters, output: VisionModelOutput) -> Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        if self.layer_id == vision_feature_layer and model_params.return_last_layer_attention:
            hidden_states, scores = self.self_attn(hidden_states, True)
            output.scores = scores 
        else:
            hidden_states = self.self_attn(hidden_states, False)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

# class CLIPEncoder(nn.Module):
#     def __init__(self, config: CLIPVisionConfig):
#         super().__init__()
#         self.layers = nn.ModuleList([CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])
    
#     def forward(self, hidden_states: Tensor) -> tuple[list[Tensor], Tensor]:
#         all_hidden_states = []
#         all_hidden_states.append(hidden_states)
#         for _, encoder_layer in enumerate(self.layers):
#             hidden_states = encoder_layer(hidden_states)
#             all_hidden_states.append(hidden_states)
#         return hidden_states, all_hidden_states

class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        embed_dim = config.hidden_size
        image_size = config.image_size
        patch_size = config.patch_size
        num_positions = (image_size // patch_size) ** 2 + 1

        self.class_embedding = nn.Parameter(torch.randn(embed_dim))
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.position_embedding = nn.Embedding(num_positions, embed_dim)
        self.register_buffer("position_ids", torch.arange(num_positions), persistent=False)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values.to(self.patch_embedding.weight.dtype))  # (batch_size, hidden_size, grid, grid)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2) # (batch_size, grid * grid, hidden_size)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1) # (batch_size, 1, hidden_size)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1) # (batch_size, grid * grid + 1, hidden_size)
        embeddings = embeddings + self.position_embedding(self.position_ids[None, :]) # (gird * grid + 1) -> (1, grid * grid + 1) -> (1, grid * grid + 1, hidden_size)
        return embeddings

# class CLIPVisionTransformer(nn.Module):
#     def __init__(self, config: CLIPVisionConfig):
#         super().__init__()
#         self.embeddings = CLIPVisionEmbeddings(config)
#         self.pre_layrnorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
#         self.encoder = CLIPEncoder(config)
#         self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
#     def forward(self, pixel_values: Tensor) -> Tensor:
#         # pixel_values (batch_size, channels, width, height)
#         hidden_states = self.embeddings(pixel_values) # (batch_size, n_tokens, hidden_size)
#         hidden_states = self.pre_layrnorm(hidden_states)
#         last_hidden_state, hidden_states = self.encoder(hidden_states)
#         pooled_output = last_hidden_state[:, 0, :]
#         pooled_output = self.post_layernorm(pooled_output)
#         return pooled_output, last_hidden_state, hidden_states

# class CLIPVisionModel(nn.Module):
#     def __init__(self, config: CLIPVisionConfig):
#         super().__init__()
#         self.vision_model = CLIPVisionTransformer(config)

#     def forward(self, pixel_values: Tensor) -> Tensor:
#         # pixel_values (n_pictures, n_channels, width, height)
#         return self.vision_model(pixel_values)

class CLIPEncoder(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.layers = nn.ModuleList([CLIPEncoderLayer(config, layer_id) for layer_id in range(config.num_hidden_layers)])
    
    def forward(self, hidden_states: Tensor, vision_feature_layer: int, model_params: VisionModelParameters) -> Tensor:
        vision_feature_layer = (vision_feature_layer + len(self.layers)) % len(self.layers)
        output = VisionModelOutput()
        for i, encoder_layer in enumerate(self.layers[:vision_feature_layer + 1]): #
            hidden_states = self.layers[i](hidden_states, vision_feature_layer, model_params, output)
            
        return hidden_states, output

class CLIPVisionTransformer(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(self, pixel_values: Tensor, vision_feature_layer: int, model_params: VisionModelParameters) -> tuple[Tensor, VisionModelOutput]:
        # pixel_values (batch_size, channels, width, height)
        hidden_states = self.embeddings(pixel_values) # (batch_size, n_tokens, hidden_size)
        hidden_states = self.pre_layrnorm(hidden_states)
        hidden_states, output = self.encoder(hidden_states, vision_feature_layer, model_params)
        return hidden_states, output

class CLIPVisionModel(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.vision_model = CLIPVisionTransformer(config)

    def forward(self, pixel_values: Tensor, vision_feature_layer: int, model_params: VisionModelParameters) -> tuple[Tensor, VisionModelOutput]:
        # pixel_values (n_pictures, n_channels, width, height)
        return self.vision_model(pixel_values, vision_feature_layer, model_params)