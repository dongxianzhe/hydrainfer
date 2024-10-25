import torch
from torch import nn
from transformers import LlavaConfig

class CLIPSdpaAttention(nn.Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()
        self.k_proj = nn.Linear(config.vision_config.hidden_size, config.vision_config.hidden_size, bias=True)
        self.v_proj = nn.Linear(config.vision_config.hidden_size, config.vision_config.hidden_size, bias=True)
        self.q_proj = nn.Linear(config.vision_config.hidden_size, config.vision_config.hidden_size, bias=True)
        self.out_proj = nn.Linear(config.vision_config.hidden_size, config.vision_config.hidden_size, bias=True)

class CLIPMLP(nn.Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.vision_config.hidden_size, config.vision_config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.vision_config.intermediate_size, config.vision_config.hidden_size, bias=True)

class CLIPEncoderLayer(nn.Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()
        self.self_attn = CLIPSdpaAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.vision_config.hidden_size, eps=config.vision_config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.vision_config.hidden_size, eps=config.vision_config.layer_norm_eps)

class CLIPEncoder(nn.Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()
        self.layers = nn.ModuleList([CLIPEncoderLayer(config) for _ in range(config.vision_config.num_hidden_layers)])

class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()
        self.embed_dim = config.vision_config.hidden_size
        self.image_size = config.vision_config.image_size
        self.patch_size = config.vision_config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.vision_config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    # def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
    #     batch_size = pixel_values.shape[0]
    #     target_dtype = self.patch_embedding.weight.dtype
    #     patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
    #     patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

    #     class_embeds = self.class_embedding.expand(batch_size, 1, -1)
    #     embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
    #     embeddings = embeddings + self.position_embedding(self.position_ids)
    #     return embeddings

class CLIPVisionTransformer(nn.Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()
        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(config.vision_config.hidden_size, eps=config.vision_config.layer_norm_eps)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.vision_config.hidden_size, eps=config.vision_config.layer_norm_eps)

class CLIPVisionModel(nn.Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()
        self.vision_model = CLIPVisionTransformer(config)

class LlavaMultiModalProjector(nn.Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()
        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

class LlamaSdpaAttention(nn.Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()
        self.q_proj = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=False)

class LlamaMLP(nn.Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.text_config.hidden_size, config.text_config.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.text_config.hidden_size, config.text_config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.text_config.intermediate_size, config.text_config.hidden_size, bias=False)

class LlamaRMSNorm(nn.Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.text_config.hidden_size))
        self.variance_epsilon = config.text_config.rms_norm_eps

    # def forward(self, hidden_states):
    #     input_dtype = hidden_states.dtype
    #     hidden_states = hidden_states.to(torch.float32)
    #     variance = hidden_states.pow(2).mean(-1, keepdim=True)
    #     hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    #     return self.weight * hidden_states.to(input_dtype)

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()
        self.self_attn = LlamaSdpaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config)
        self.post_attention_layernorm = LlamaRMSNorm(config)

class LlamaModel(nn.Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.text_config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.text_config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config)

class LlamaForCausalLM(nn.Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.vocab_size, bias=False)

class LlavaForConditionalGeneration(nn.Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()
        self.vision_tower = CLIPVisionModel(config)
        self.multi_modal_projector = LlavaMultiModalProjector(config)
        self.language_model = LlamaForCausalLM(config)