from dataclasses import dataclass
import torch
from torch import nn, Tensor
import os
import safetensors.torch
from jinja2 import Template
from typing import Optional
from transformers import AutoTokenizer, AutoProcessor
from hydrainfer.layer.norm import RMSNorm
from hydrainfer.model.model_forward import UpDownMLP, GateUpDownMLP, ROPECausalGroupedQueryPageAttention, DecoderLayer
from hydrainfer.model.parameters import VisionModelParameters, VisionModelOutput, LanguageModelOutput, LanguageModelParameters
from hydrainfer.layer.rotary_embedding import RotaryEmbedding, compute_default_inv_freq
from hydrainfer.layer.causal_attention import AttentionParameters
from hydrainfer.layer.multihead_attention import MultiHeadAttention, MultiHeadAttentionConfig, MultiHeadAttentionOutput, MultiHeadAttentionParameters
from hydrainfer.model.model_factory import ModelFactory, ModelFactoryConfig, ModelFactoryContext, VisionModel, LanguageModel, VisionModelConfig, LanguageModelConfig, Tokenizer, ImageTokenCaculator
from hydrainfer.utils.torch_utils import str2device, str2dtype
from hydrainfer.model.model_profiler import ModelProfiler, ModelParamsConfig, VisionLanguageModelProfiler
from hydrainfer.model.model_loader import load_config_from_json, load_safetensor

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
    num_attention_heads: int

@dataclass
class InternLM2ForCausalLMConfig:
    hidden_size: int
    intermediate_size: int
    rms_norm_eps: float
    vocab_size: int
    num_hidden_layers: int
    hidden_act: str
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rope_theta: float
    eos_token_id: int


@dataclass
class InternVLChatModelConfig:
    llm_config: InternLM2ForCausalLMConfig
    vision_config: InternVisionModelConfig
    select_layer: int
    ps_version: str
    downsample_ratio: float


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
        self.config = config
        self.head_dim = config.hidden_size // config.num_attention_heads 
        self.qkv = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size * 3, bias=False)
        self.attn_drop = nn.Dropout(p=0.0, inplace=False)
        self.proj_drop = nn.Dropout(p=0.0, inplace=False)
        self.q_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.layer_norm_eps)
        self.k_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.layer_norm_eps)
        self.proj = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=True)

        self.attention = MultiHeadAttention(MultiHeadAttentionConfig(config.num_attention_heads, self.head_dim))

    def forward(self, hidden_states: Tensor) -> Tensor:
        # hidden_states (batch_size, num_tokens_per_image, hidden_size)
        qkv = self.qkv(hidden_states)
        query = qkv[:self.config.hidden_size]
        key   = qkv[self.config.hidden_size:2*self.config.hidden_size]
        value = qkv[2*self.config.hidden_size:]
        attention_output = self.attention(query, key, value, MultiHeadAttentionParameters(return_scores=False))
        o = attention_output.o # (batch_size, num_tokens_per_image, hidden_size)
        o = self.out_proj(o) # (batch_size, num_tokens_per_image, hidden_size)
        return o

class InternMLP(nn.Module):
    def __init__(self, config: InternVisionModelConfig):
        super().__init__()
        assert config.hidden_act == "gelu"
        self.act = nn.GELU()
        self.fc1 = nn.Linear(in_features=config.hidden_size, out_features=config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(in_features=config.intermediate_size, out_features=config.hidden_size, bias=True)

        self.mlp = UpDownMLP(up_proj=self.fc1, down_proj=self.fc2, activation=self.fc1)

    def forward(self, h: Tensor) -> Tensor:
        return self.mlp.forward(h)


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


class InternLM2Attention(nn.Module):
    def __init__(self, config: InternLM2ForCausalLMConfig):
        super().__init__()
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.wqkv = nn.Linear(in_features=config.hidden_size, out_features=(config.num_attention_heads + 2 * config.num_key_value_heads) * self.head_dim, bias=False)
        self.wo = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(
            rotary_dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            inv_freq=compute_default_inv_freq(
                rotary_dim=config.head_dim,
                theta=config.rope_theta
                ),
            interleaved=False
            ) # todo rope scaling
        self.attention = ROPECausalGroupedQueryPageAttention(qkv_proj=self.wqkv, o_proj=self.wo, rotary_emb=self.rotary_emb, n_qo_heads=config.num_attention_heads, n_kv_heads=config.num_key_value_heads, head_dim=self.head_dim, )

    def forward(self, hidden_states: Tensor, position_ids: Tensor, attention_param: AttentionParameters) -> Tensor:
        return self.attention(hidden_states, position_ids, attention_param)

class InternLM2MLP(nn.Module):
    def __init__(self, config: InternLM2ForCausalLMConfig):
        super().__init__()
        self.w1 = nn.Linear(in_features=config.hidden_size, out_features=config.intermediate_size, bias=False)
        self.w3 = nn.Linear(in_features=config.hidden_size, out_features=config.intermediate_size, bias=False)
        self.w2 = nn.Linear(in_features=config.intermediate_size, out_features=config.hidden_size, bias=False)
        assert config.hidden_act == 'silu'
        self.act_fn = nn.SiLU()
        self.mlp = GateUpDownMLP(gate_proj=self.w1, up_proj=self.w3, down_proj=self.w2, activation=self.act_fn)
        
    def forward(self, h: Tensor) -> Tensor:
        return self.mlp.forward(h)


class InternLM2DecoderLayer(nn.Module):
    def __init__(self, config: InternLM2ForCausalLMConfig, layer_id: int):
        super().__init__()
        self.attention = InternLM2Attention(config)
        self.feed_forward = InternLM2MLP(config)
        self.attention_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)

        self.decoder_layer = DecoderLayer(attention=self.attention, mlp=self.feed_forward, norm_1=self.attention_norm, norm_2=self.ffn_norm, layer_id=layer_id, n_layers=config.num_hidden_layers)

    def forward(self, hidden_states: Tensor, position_ids: Tensor, model_params: LanguageModelParameters) -> Tensor:
        return self.decoder_layer.forward(hidden_states, position_ids, model_params)

class InternLM2Model(nn.Module):
    def __init__(self, config: InternLM2ForCausalLMConfig):
        super().__init__()
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=2)
        self.layers = nn.ModuleList([InternLM2DecoderLayer(config, layer_id) for layer_id in range(config.num_hidden_layers)])
        self.norm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)


    def forward(self, input_ids: Tensor, position_ids: Tensor, model_params: LanguageModelParameters) -> Tensor:
        if input_ids.dtype == torch.int:
            hidden_states = self.tok_embeddings(input_ids)
        else:
            hidden_states = input_ids

        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, position_ids, model_params)
        return self.norm(hidden_states)


class InternLM2ForCausalLM(nn.Module):
    def __init__(self, config: InternLM2ForCausalLMConfig):
        super().__init__()
        self.model = InternLM2Model(config)
        self.output = nn.Linear(in_features=config.hidden_size, out_features=config.vocab_size, bias=False)

    def forward(self, input_ids_or_input_embeds: Tensor, position_ids: Tensor, model_params: LanguageModelParameters) -> Tensor:
        # input_ids (n_tokens, ) or input embeds (n_tokens, hidden_size)
        hidden_state = self.model(input_ids_or_input_embeds, position_ids, model_params) # hidden_state (n_selected_tokens, hidden_size) we discard tokens that do not need to be sampled before entering into the last ffn layer to reduce redundant computation
        logits = self.lm_head(hidden_state) # (n_selected_tokens, hidden_size)
        sample_token_ids = torch.argmax(logits, dim=-1, keepdim=False) # (n_selected_tokens, )
        return sample_token_ids


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


class InternVLVisionModel(VisionModel):
    def __init__(self, model_path: str, dtype: torch.dtype, device: torch.device):
        super().__init__()
        # 1. config
        config = load_config_from_json(InternVLChatModelConfig, model_path)
        self.vision_feature_layer = config.select_layer
        self.h = self.w = config.vision_config.image_size // config.vision_config.patch_size
        # 2. create model
        torch.set_default_dtype(dtype)
        with torch.device(device):
            self.vision_tower = InternVisionModel(config.vision_config)
            self.multi_modal_projector = nn.Sequential(
                nn.LayerNorm(config.vision_config.intermediate_size, eps=config.llm_config.rms_norm_eps, elementwise_affine=True), 
                nn.Linear(in_features=config.vision_config.intermediate_size, out_features=config.llm_config.hidden_size, bias=True), 
                nn.GELU(approximate='none'), 
                nn.Linear(in_features=config.llm_config.hidden_size, out_features=config.llm_config.hidden_size, bias=True), 
            )
        torch.set_default_dtype(torch.float)
        # 3. load vision_tower state dict
        load_safetensor(
            model_with_prefix_list=[
                (self.vision_tower, 'vision_model.'), 
                (self.multi_modal_projector, 'mlp1.'), 
            ], 
            param_with_name_list=[], 
            model_weights_path=model_path, 
        )

        self.vision_tower.to(dtype)
        self.vision_tower.eval()
        self.multi_modal_projector.to(dtype)
        self.multi_modal_projector.eval()

    def pixel_shuffle(self, x: Tensor, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor), int(c / (scale_factor * scale_factor)))
        assert config.ps_version == 'v2'
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    
    def forward(self, pixel_values: list[Tensor], model_params: VisionModelParameters) -> VisionModelOutput:
        pixel_values = torch.cat(pixel_values, dim=0)
        assert pixel_values.dim() == 4, f'pixel value shape should be 4 dim but got {pixel_values.shape}'
        # pixel_values (n_images, n_channels, height, width)
        output = self.vision_tower(pixel_values, self.vision_feature_layer, model_params) # (n_images, num_tokens_per_images, hidden_size of vision model)
        vit_embeds = output.image_features
        vit_embeds = vit_embeds[:, 1:] # (n_images, num_tokens_per_images - 1, hidden_size of vision model) evict first class token of each image

        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], self.h, self.w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=config.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)

        image_features = self.multi_modal_projector(vit_embeds) # (n_images, (num_tokens_per_images - 1) * downsample_ratio * downsample_ratio, hidden_size of language model)
        output.image_features = image_features
        return output


class InternVLLanguageModel(LanguageModel):
    def __init__(self, model_path: str, dtype: torch.dtype, device: torch.device):
        super().__init__()
        # 1. config
        config = load_config_from_json(InternVLChatModelConfig, self.path)
        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMG_END_TOKEN)
        # 2. create model
        torch.set_default_dtype(dtype)
        with torch.device(device):
            self.language_model = InternLM2ForCausalLM(config.text_config)
        torch.set_default_dtype(torch.float)
        # 3. load vision_tower state dict
        load_safetensor(model_with_prefix_list=[(self.language_model, 'language_model.')], param_with_name_list=[], model_weights_path=model_path)

        # to ensure that all tensor data type are correct such as non persistent rope inv freq, becuase it is created with float dtype specifically and will not be affected by set_default_dtype
        self.language_model.to(dtype)
        self.language_model.eval()
    
    def forward(self, input_ids: Tensor, image_features: Optional[Tensor], position_ids: Tensor, model_params: LanguageModelParameters) -> LanguageModelOutput:
        # input_ids      (n_text_tokens + n_image_tokens) n_text_tokens is number of text tokens, n_image_tokens is number of image tokens
        # image_features (n_text_tokens, hidden_size)
        # position_ids (n_text_tokens + n_image_tokens)
        input_embeds = self.language_model.model.embed_tokens(input_ids)
        if image_features is not None:
            image_overwrite_mask = input_ids == self.image_token_id
            input_embeds[image_overwrite_mask, :] = image_features.view(-1, input_embeds.shape[-1])
        sample_token_ids = self.language_model(input_embeds, position_ids, model_params) # (n_selected_tokens, )

        return LanguageModelOutput(
            sample_token_ids = sample_token_ids,
        )


class InternVLTokenizer(Tokenizer):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id
        current_script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(current_script_path)
        template_path = os.path.join(script_dir, "chat_template", "template_llava.jinja")
        with open(template_path, 'r', encoding='utf-8') as file:
            self.chat_template = Template(file.read())

    def encode(self, prompt: str) -> list[int]:
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        return token_ids

    def decode(self, token_id: int) -> str:
        # note that this is not an ASCII U+005F _, it is U+2581
        if self.tokenizer.convert_ids_to_tokens([token_id])[0].startswith('▁'):
            text = " " + self.tokenizer.decode([token_id])
        else:
            text = self.tokenizer.decode([token_id])
        return text

    def apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        prompt = self.chat_template.render(
            messages = messages, 
            bos_token = self.bos_token, 
            eos_token = self.eos_token,
            add_generation_prompt = True, 
        )
        return prompt


class InternVLTokenCaculator(ImageTokenCaculator):
    def __init__(self, n_tokens_per_image: int):
        self.n_tokens_per_image = n_tokens_per_image
    def get_num_image_tokens(self, image_size: tuple[int, int]) -> int:
        return self.n_tokens_per_image
        

class InternVLModelFactory(ModelFactory):
    def __init__(self, config: ModelFactoryConfig, context: ModelFactoryContext):
        self.path = config.path
        self.dtype = str2dtype(config.dtype)
        self.device = str2device(config.device)

    def getVisionModel(self) -> VisionModel:
        return InternVLVisionModel(self.path, self.dtype, self.device)

    def getLanguageModel(self) -> LanguageModel:
        return InternVLLanguageModel(self.path, self.dtype, self.device)

    def getVisionModelConfig(self) -> VisionModelConfig:
        config_ref = load_config_from_json(InternVLChatModelConfig, self.path)
        n_tokens_per_image = int((config.vision_config.image_size // config.vision_config.patch_size) ** 2 * (config.downsample_ratio ** 2))
        IMG_START_TOKEN = '<img>'
        IMG_END_TOKEN = '</img>',
        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        image_token_id = self.tokenizer.convert_tokens_to_ids(self.IMG_END_TOKEN)
        return VisionModelConfig(
            image_token = IMG_START_TOKEN + IMG_CONTEXT_TOKEN + IMG_END_TOKEN, 
            image_token_id = image_token_id, 
            image_token_caculator = InternVLTokenCaculator(n_tokens_per_image=n_tokens_per_image),
        )

    def getLanguageModelConfig(self) -> LanguageModelConfig:
        config_ref = load_config_from_json(InternVLChatModelConfig, self.path)
        return LanguageModelConfig(
            n_layers = config_ref.llm_config.num_hidden_layers, 
            max_position_embeddings = config_ref.llm_config.max_position_embeddings, 
            n_qo_heads = config_ref.llm_config.num_attention_heads, 
            n_kv_heads = config_ref.llm_config.num_key_value_heads, 
            head_dim = config_ref.llm_config.hidden_size // config_ref.llm_config.num_attention_heads, 
            eos_token_id = config_ref.llm_config.eos_token_id, 
        )

    def getProcessor(self) -> AutoProcessor:
        return AutoProcessor.from_pretrained(self.path, trust_remote_code=True)

    def getTokenizer(self) -> Tokenizer:
        return InternVLTokenizer(self.path)

    def getModelProfiler(self) -> ModelProfiler:
        return VisionLanguageModelProfiler(self.path, vision_model_prefixes=['vision_model.', 'mlp1.'], language_model_prefixes=['language_model.'])


if __name__ == '__main__':
    model_weights_path = "/models/OpenGVLab/InternVL2-26B"
    config = load_config_from_json(config_class=InternVLChatModelConfig, model_weights_path=model_weights_path)
    # print(config)

    # torch.set_default_device(torch.device('cuda:0'))
    # torch.set_default_dtype(torch.half)
    # model = InternVLChatModel(config)
    # load_safetensor_strict_equal(model, model_weights_path=model_weights_path)
    # # load_safetensor(model_with_prefix_list=[(model, '')], param_with_name_list=[], model_weights_path=model_weights_path)
    from hydrainfer.model.model_factory import getModelFactory
    factory = getModelFactory(ModelFactoryConfig(path=model_weights_path), ModelFactoryContext())
    tokenizer = factory.getTokenizer()
    token_ids = tokenizer.encode('<img><IMG_CONTEXT></img> hello world')
    text = ""
    for token_id in token_ids:
        text += tokenizer.decode(token_id)
    print(f'token_ids {token_ids}')
    print(f'text {text}')

    # IMG_START_TOKEN='<img>'
    # IMG_END_TOKEN='</img>',
    # IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'
    # print(f'tokenizer.encode(IMG_START_TOKEN) {tokenizer.encode(IMG_START_TOKEN)}')
    # print(f'tokenizer.encode(IMG_END_TOKEN) {tokenizer.encode(IMG_END_TOKEN)}')
    # print(f'tokenizer.encode(IMG_CONTEXT_TOKEN) {tokenizer.encode(IMG_CONTEXT_TOKEN)}')

    # messages = [
    #     {"role": "user", "content": "Hello!"},
    #     {"role": "assistant", "content": "Hi there!"},
    #     {"role": "user", "content": "How are you?"},
    # ]
    # print(tokenizer.apply_chat_template(messages))

    language_config = factory.getLanguageModelConfig()
    vision_model = factory.getVisionModel()