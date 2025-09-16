import torch
from torch import nn, Tensor
from transformers import CLIPVisionConfig
from hydrainfer.model.parameters import VisionModelParameters, VisionModelOutput
from hydrainfer.layer.activation import QuickGELUActivation
from hydrainfer.layer.multihead_attention import MultiHeadAttentionConfig, MultiHeadAttentionParameters, MultiHeadAttention
from hydrainfer.model.model_forward import UpDownMLP


class CLIPSdpaAttention(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.n_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads 
        self.attention = MultiHeadAttention(MultiHeadAttentionConfig(self.n_heads, self.head_dim))
    
    def forward(self, hidden_states: Tensor, return_scores: bool = False) -> tuple[Tensor, Tensor]:
        # hidden_states (batch_size, num_tokens_per_image, hidden_size)
        query = self.q_proj(hidden_states) # query (batch_size, num_tokens_per_image, hidden_size)
        key   = self.k_proj(hidden_states) # key   (batch_size, num_tokens_per_image, hidden_size)
        value = self.v_proj(hidden_states) # value (batch_size, num_tokens_per_image, hidden_size)
        attention_output = self.attention(query, key, value, MultiHeadAttentionParameters(return_scores=return_scores))
        o = attention_output.o # (batch_size, num_tokens_per_image, hidden_size)
        scores = attention_output.attention_scores
        o = self.out_proj(o) # (batch_size, num_tokens_per_image, hidden_size)
        return o, scores


class CLIPMLP(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.activation_fn = QuickGELUActivation()
        self.mlp = UpDownMLP(up_proj = self.fc1, down_proj = self.fc2, activation = QuickGELUActivation())
        
    
    def forward(self, h: Tensor) -> Tensor:
        # hidden_states (batch_size, num_tokens_per_image, hidden_size)
        return self.mlp.forward(h)


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
        hidden_states, scores = self.self_attn(hidden_states, self.layer_id == vision_feature_layer and model_params.return_last_layer_attention)
        output.scores = scores 
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


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
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) # this layer is used in train but not used in inference
    
    def forward(self, pixel_values: Tensor, vision_feature_layer: int, model_params: VisionModelParameters) -> tuple[Tensor, VisionModelOutput]:
        # pixel_values (batch_size, channels, height, width)
        hidden_states = self.embeddings(pixel_values) 
        # hidden_states (batch_size, num_token_per_image, hidden_size)
        hidden_states = self.pre_layrnorm(hidden_states)
        # hidden_states (batch_size, num_token_per_image, hidden_size)
        hidden_states, output = self.encoder(hidden_states, vision_feature_layer, model_params)
        # hidden_states (batch_size, num_token_per_image, hidden_size)
        return hidden_states, output


class CLIPVisionModel(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.vision_model = CLIPVisionTransformer(config)

    def forward(self, pixel_values: Tensor, vision_feature_layer: int, model_params: VisionModelParameters) -> tuple[Tensor, VisionModelOutput]:
        # pixel_values (batch_size, n_channels, height, width)
        return self.vision_model(pixel_values, vision_feature_layer, model_params)