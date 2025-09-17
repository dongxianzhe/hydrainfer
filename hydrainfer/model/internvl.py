from dataclasses import dataclass
import torch
from torch import nn, Tensor
import os
import safetensors.torch
from hydrainfer.layer.norm import RMSNorm
from hydrainfer.model.model_forward import UpDownMLP
from hydrainfer.model.parameters import VisionModelParameters, VisionModelOutput


def load_safetensor_strict_equal(model: nn.Module, model_weights_path: str):
    state_dict = model.state_dict()
    loaded_set = set()
    for entry in os.scandir(model_weights_path):
        if entry.is_file() and os.path.splitext(entry.name)[1] == '.safetensors':
            for name, weight in safetensors.torch.load_file(entry.path).items():
                print(f'{name}, {weight.shape}, {state_dict[name].shape}')
                state_dict[name].data.copy_(weight)
                loaded_set.add(name)
    assert len(state_dict) == len(loaded_set), f'expected load {len(state_dict)} tensors, but only {len(loaded_set)} tensors are loaded, {state_dict.keys() - loaded_set} are not loaded'


@dataclass
class InternVisionModelConfig:
    hidden_size: int
    intermediate_size: int
    initializer_factor: float
    layer_norm_eps: float
    num_hidden_layers: int
    drop_path_rate: float
    hidden_act: str
    patch_size: int
    image_size: int
    num_channels: int

@dataclass
class InternLM2ForCausalLMConfig:
    hidden_size: int
    intermediate_size: int
    rms_norm_eps: float
    vocab_size: int
    num_hidden_layers: int


@dataclass
class InternVLChatModelConfig:
    llm_config: InternLM2ForCausalLMConfig
    vision_config: InternVisionModelConfig


class InternVisionEmbeddings(nn.Module):
    def __init__(self, config: InternVisionModelConfig):
        super().__init__()
        self.patch_embedding = nn.Conv2d(in_channels=config.num_channels, out_channels=config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size)
        self.class_embedding = nn.Parameter(torch.randn(1, 1, config.hidden_size))

        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_positions, config.hidden_size))

    def _get_pos_embed(self, pos_embed: Tensor, H: int, W: int) -> Tensor:
        target_dtype = pos_embed.dtype
        pos_embed = pos_embed.float().reshape(
            1, self.image_size // self.patch_size, self.image_size // self.patch_size, -1).permute(0, 3, 1, 2)
        pos_embed = nn.functional.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False). \
            reshape(1, -1, H * W).permute(0, 2, 1).to(target_dtype)
        return pos_embed

    def forward(self, pixel_values: Tensor) -> Tensor:
        # pixel_values (batch_size, channel, width, height)
        patch_embeds = self.patch_embedding(pixel_values) # patch_embeds (batch_size, hidden_size, width / patch_size, height / patch_size)
        batch_size, _, height, width = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2) # (batch_size, width / patch_size * height / patch_size, hidden_size)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(pixel_values.dtype)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1) # embeddings (batch_size, num_positions, hidden_size)
        position_embedding = torch.cat([
            self.position_embedding[:, :1, :],
            self._get_pos_embed(self.position_embedding[:, 1:, :], height, width)
        ], dim=1)
        return embeddings + position_embedding.to(pixel_values.dtype)

class InternAttention(nn.Module):
    def __init__(self, config: InternVisionModelConfig):
        super().__init__()
        self.qkv = nn.Linear(in_features=config.hidden_size, out_features=9600, bias=False)
        self.attn_drop = nn.Dropout(p=0.0, inplace=False)
        self.proj_drop = nn.Dropout(p=0.0, inplace=False)
        self.q_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.layer_norm_eps)
        self.k_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.layer_norm_eps)
        self.proj = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=True)


class InternMLP(nn.Module):
    def __init__(self, config: InternVisionModelConfig):
        super().__init__()
        assert config.hidden_act == "gelu"
        self.act = nn.GELU()
        self.fc1 = nn.Linear(in_features=config.hidden_size, out_features=config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(in_features=config.intermediate_size, out_features=config.hidden_size, bias=True)

        self.mlp = UpDownMLP(up_proj=self.fc1, down_proj=self.fc2, activation=self.fc1)

    def forward(self, h: Tensor) -> Tensor:
        return self.mlp(h)


class InternVisionEncoderLayer(nn.Module):
    def __init__(self, config: InternVisionModelConfig):
        super().__init__()
        self.attn = InternAttention(config)
        self.mlp = InternMLP(config)
        self.norm1 = RMSNorm(hidden_size=config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = RMSNorm(hidden_size=config.hidden_size, eps=config.layer_norm_eps)
        self.ls1 = nn.Parameter(config.initializer_factor * torch.ones(config.hidden_size))
        self.ls2 = nn.Parameter(config.initializer_factor * torch.ones(config.hidden_size))
        assert config.drop_path_rate == 0.0
        self.drop_path1 = nn.Identity()
        self.drop_path2 = nn.Identity()

    def forward(self, hidden_states: Tensor, vision_feature_layer: int, model_params: VisionModelParameters) -> Tensor:
        hidden_states = hidden_states + self.drop_path1(self.attn(self.norm1(hidden_states)) * self.ls1)
        hidden_states = hidden_states + self.drop_path2(self.mlp(self.norm2(hidden_states)) * self.ls2)
        return hidden_states


class InternVisionEncoder(nn.Module):
    def __init__(self, config: InternVisionModelConfig):
        super().__init__()
        self.layers = nn.ModuleList([InternVisionEncoderLayer(config) for i in range(config.num_hidden_layers)])

    def forward(self, hidden_states: Tensor, vision_feature_layer: int, model_params: VisionModelParameters) -> Tensor:
        vision_feature_layer = (vision_feature_layer + len(self.layers)) % len(self.layers)
        for layer_id, layer in enumerate(self.layers[:vision_feature_layer+1]):
            hidden_states = layer(hidden_states)
        return hidden_states


class InternVisionModel(nn.Module):
    def __init__(self, config: InternVisionModelConfig):
        super().__init__()
        self.embeddings = InternVisionEmbeddings(config)
        self.encoder = InternVisionEncoder(config)

    def forward(self, pixel_values: Tensor, vision_feature_layer: int, model_params: VisionModelParameters) -> VisionModelOutput:
        h = self.embeddings(pixel_values)
        h = self.encoder(h)
        return VisionModelOutput(image_features=h)


class InternLM2DynamicNTKScalingRotaryEmbedding(nn.Module):
    def __init__(self, config: InternLM2ForCausalLMConfig):
        super().__init__()


class InternLM2Attention(nn.Module):
    def __init__(self, config: InternLM2ForCausalLMConfig):
        super().__init__()
        self.wqkv = nn.Linear(in_features=config.hidden_size, out_features=8192, bias=False)
        self.wo = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)
        self.rotary_emb = InternLM2DynamicNTKScalingRotaryEmbedding(config)


class InternLM2MLP(nn.Module):
    def __init__(self, config: InternLM2ForCausalLMConfig):
        super().__init__()
        self.w1 = nn.Linear(in_features=config.hidden_size, out_features=config.intermediate_size, bias=False)
        self.w3 = nn.Linear(in_features=config.hidden_size, out_features=config.intermediate_size, bias=False)
        self.w2 = nn.Linear(in_features=config.intermediate_size, out_features=config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()


class InternLM2DecoderLayer(nn.Module):
    def __init__(self, config: InternLM2ForCausalLMConfig):
        super().__init__()
        self.attention = InternLM2Attention(config)
        self.feed_forward = InternLM2MLP(config)
        self.attention_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)


class InternLM2Model(nn.Module):
    def __init__(self, config: InternLM2ForCausalLMConfig):
        super().__init__()
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=2)
        self.layers = nn.ModuleList([InternLM2DecoderLayer(config) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)


class InternLM2ForCausalLM(nn.Module):
    def __init__(self, config: InternLM2ForCausalLMConfig):
        super().__init__()
        self.model = InternLM2Model(config)
        self.output = nn.Linear(in_features=config.hidden_size, out_features=config.vocab_size, bias=False)


class InternVLChatModel(nn.Module):
    def __init__(self, config: InternVLChatModelConfig):
        super().__init__()
        self.vision_model = InternVisionModel(config.vision_config)
        self.language_model = InternLM2ForCausalLM(config.llm_config)
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(config.vision_config.intermediate_size, eps=config.llm_config.rms_norm_eps, elementwise_affine=True), 
            nn.Linear(in_features=config.vision_config.intermediate_size, out_features=config.llm_config.hidden_size, bias=True), 
            nn.GELU(approximate='none'), 
            nn.Linear(in_features=config.llm_config.hidden_size, out_features=config.llm_config.hidden_size, bias=True), 
        )


if __name__ == '__main__':
    from hydrainfer.model.model_loader import load_config_from_json, load_safetensor
    model_weights_path = "/models/OpenGVLab/InternVL2-26B"
    config = load_config_from_json(config_class=InternVLChatModelConfig, model_weights_path=model_weights_path)
    print(config)
    torch.set_default_device(torch.device('cuda:0'))
    torch.set_default_dtype(torch.half)
    model = InternVLChatModel(config)
    load_safetensor_strict_equal(model, model_weights_path=model_weights_path)
    # load_safetensor(model_with_prefix_list=[(model, '')], param_with_name_list=[], model_weights_path=model_weights_path)