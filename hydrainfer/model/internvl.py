from dataclasses import dataclass
import torch
from torch import nn
import os
import safetensors.torch
from hydrainfer.layer.norm import RMSNorm


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
class InternVLChatModelConfig:
    pass


class InternVisionEmbeddings(nn.Module):
    def __init__(self, config: InternVLChatModelConfig):
        super().__init__()
        self.patch_embedding = nn.Conv2d(in_channels=3, out_channels=3200, kernel_size=14, stride=14)
        self.class_embedding = nn.Parameter(torch.randn(1, 1, 3200))

        self.image_size = config.image_size = 448
        self.patch_size = config.patch_size = 14
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_positions, 3200))


class InternAttention(nn.Module):
    def __init__(self, config: InternVLChatModelConfig):
        super().__init__()
        self.qkv = nn.Linear(in_features=3200, out_features=9600, bias=False)
        self.attn_drop = nn.Dropout(p=0.0, inplace=False)
        self.proj_drop = nn.Dropout(p=0.0, inplace=False)
        self.q_norm = RMSNorm(hidden_size=3200, eps=1e-6)
        self.k_norm = RMSNorm(hidden_size=3200, eps=1e-6)
        self.proj = nn.Linear(in_features=3200, out_features=3200, bias=True)


class GELUActivation(nn.Module):
    def __init__(self, config: InternVLChatModelConfig):
        super().__init__()


class InternMLP(nn.Module):
    def __init__(self, config: InternVLChatModelConfig):
        super().__init__()
        self.act = GELUActivation(config)
        self.fc1 = nn.Linear(in_features=3200, out_features=12800, bias=True)
        self.fc2 = nn.Linear(in_features=12800, out_features=3200, bias=True)


class Identity(nn.Module):
    def __init__(self, config: InternVLChatModelConfig):
        super().__init__()


class InternVisionEncoderLayer(nn.Module):
    def __init__(self, config: InternVLChatModelConfig):
        super().__init__()
        self.attn = InternAttention(config)
        self.mlp = InternMLP(config)
        self.norm1 = RMSNorm(hidden_size=3200, eps=1e-6)
        self.norm2 = RMSNorm(hidden_size=3200, eps=1e-6)
        # self.ls1 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))
        # self.ls2 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))
        self.ls1 = nn.Parameter(0.1 * torch.ones(3200))
        self.ls2 = nn.Parameter(0.1 * torch.ones(3200))
        self.drop_path1 = Identity(config)
        self.drop_path2 = Identity(config)


class InternVisionEncoder(nn.Module):
    def __init__(self, config: InternVLChatModelConfig):
        super().__init__()
        self.layers = nn.ModuleList([InternVisionEncoderLayer(config) for i in range(45)])


class InternVisionModel(nn.Module):
    def __init__(self, config: InternVLChatModelConfig):
        super().__init__()
        self.embeddings = InternVisionEmbeddings(config)
        self.encoder = InternVisionEncoder(config)


class InternLM2DynamicNTKScalingRotaryEmbedding(nn.Module):
    def __init__(self, config: InternVLChatModelConfig):
        super().__init__()


class InternLM2Attention(nn.Module):
    def __init__(self, config: InternVLChatModelConfig):
        super().__init__()
        self.wqkv = nn.Linear(in_features=6144, out_features=8192, bias=False)
        self.wo = nn.Linear(in_features=6144, out_features=6144, bias=False)
        self.rotary_emb = InternLM2DynamicNTKScalingRotaryEmbedding(config)


class InternLM2MLP(nn.Module):
    def __init__(self, config: InternVLChatModelConfig):
        super().__init__()
        self.w1 = nn.Linear(in_features=6144, out_features=16384, bias=False)
        self.w3 = nn.Linear(in_features=6144, out_features=16384, bias=False)
        self.w2 = nn.Linear(in_features=16384, out_features=6144, bias=False)
        self.act_fn = nn.SiLU()


class InternLM2DecoderLayer(nn.Module):
    def __init__(self, config: InternVLChatModelConfig):
        super().__init__()
        self.attention = InternLM2Attention(config)
        self.feed_forward = InternLM2MLP(config)
        self.attention_norm = RMSNorm(hidden_size=6144, eps=1e-5)
        self.ffn_norm = RMSNorm(hidden_size=6144, eps=1e-5)


class InternLM2Model(nn.Module):
    def __init__(self, config: InternVLChatModelConfig):
        super().__init__()
        self.tok_embeddings = nn.Embedding(92553, 6144, padding_idx=2)
        self.layers = nn.ModuleList([InternLM2DecoderLayer(config) for i in range(48)])
        self.norm = RMSNorm(hidden_size=6144, eps=1e-5)


class InternLM2ForCausalLM(nn.Module):
    def __init__(self, config: InternVLChatModelConfig):
        super().__init__()
        self.model = InternLM2Model(config)
        self.output = nn.Linear(in_features=6144, out_features=92553, bias=False)


class InternVLChatModel(nn.Module):
    def __init__(self, config: InternVLChatModelConfig):
        super().__init__()
        self.vision_model = InternVisionModel(config)
        self.language_model = InternLM2ForCausalLM(config)
        self.mlp1 = nn.Sequential(
            nn.LayerNorm((12800,), eps=1e-05, elementwise_affine=True), 
            nn.Linear(in_features=12800, out_features=6144, bias=True), 
            nn.GELU(approximate='none'), 
            nn.Linear(in_features=6144, out_features=6144, bias=True), 
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