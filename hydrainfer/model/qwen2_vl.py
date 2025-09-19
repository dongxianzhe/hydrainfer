import os
import math
import torch
import safetensors.torch
from jinja2 import Template
from typing import Optional
from torch import nn, Tensor
from transformers import Qwen2VLConfig, AutoProcessor, AutoTokenizer
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel,PatchEmbed,PatchMerger
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig
from hydrainfer.layer.rotary_embedding import RotaryEmbedding, compute_default_inv_freq
from hydrainfer.model.model_profiler import VisionLanguageModelProfiler
from hydrainfer.model.downloader import download_hf_model
from hydrainfer.model.parameters import AttentionParameters, LanguageModelParameters, LanguageModelOutput, VisionModelParameters, VisionModelOutput
from hydrainfer.model.model_factory import VisionModel, VisionModelConfig, LanguageModel, LanguageModelConfig, ModelFactory, ModelFactoryConfig, ModelFactoryContext, ImageTokenCaculator, Tokenizer, ModelProfiler, ImageProcessor
from hydrainfer.layer.causal_attention import CausalGroupedQueryPageAttention, CausalGroupedQueryPageAttentionConfig
from hydrainfer.layer.norm import RMSNorm
from hydrainfer.layer.activation import silu
from hydrainfer.layer.multihead_attention import MultiHeadAttentionConfig,QwenMultiHeadAttention
from hydrainfer.utils.torch_utils import str2dtype, str2device
from functools import partial
from hydrainfer.utils.logger import getLogger
from hydrainfer.model.model_loader import load_safetensor
from hydrainfer.layer.activation import Silu
from hydrainfer.model.model_forward import GateUpDownMLP, UpDownMLP, ROPECausalGroupedQueryPageAttention, DecoderLayer
from hydrainfer.model.processor import TransformersAutoProcessorAdapter
logger = getLogger(__name__)

smart_resize = partial(smart_resize, max_pixels=3584 * 3584)

class Qwen2VLImageTokenCaculator(ImageTokenCaculator):
    def get_num_image_tokens(self, image_size: tuple[int, int]) -> int:
        """
        image_size (height, width)
        """
        height, width = image_size
        height, width = smart_resize(height,width)
        return (height // 14) * (width // 14) // 4

class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype) # (seq_len, )
        freqs = torch.outer(seq, self.inv_freq) # (seq_len, head_dim // 4)
        return freqs


class VisionMlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, hidden_act: str) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = Silu()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.mlp = UpDownMLP(up_proj=self.fc1, down_proj=self.fc2, activation=self.act)

    def forward(self, h: Tensor) -> torch.Tensor:
        return self.mlp.forward(h)

class VisionAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim*3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attention = QwenMultiHeadAttention(MultiHeadAttentionConfig(self.num_heads, self.head_dim))

    def rotate_half(self, x: Tensor):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb_vision(self, tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        orig_dtype = tensor.dtype
        tensor = tensor.float()
        cos = freqs.cos()
        sin = freqs.sin()
        cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
        sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
        output = (tensor * cos) + (self.rotate_half(tensor) * sin)
        output = output.to(orig_dtype)
        return output

    def forward(self, hidden_states: Tensor, cu_seqlens: Tensor, rotary_pos_emb: Tensor) -> Tensor:
        # hidden_states (total_patches, vision_model_hidden_size)
        # cu_seqlens (total_times + 1, )
        # rotary_pos_emb (total_patches, head_dim // 2)
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        # q k v (total_patches, n_heads, head_dim)
        q = self.apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = self.apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)
        attn_output = self.attention(q, k, v, seq_length, cu_seqlens)
        attn_output = self.proj(attn_output)
        return attn_output

class Qwen2VLVisionBlock(nn.Module):
    def __init__(self, config: Qwen2VLVisionConfig, attn_implementation: str):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.embed_dim, eps=1e-6)
        mlp_hidden_dim = int(config.embed_dim * config.mlp_ratio)
        self.attn = VisionAttention(config.embed_dim,num_heads=config.num_heads)
        self.mlp = VisionMlp(dim=config.embed_dim, hidden_dim=mlp_hidden_dim, hidden_act=config.hidden_act)

    def forward(self, hidden_states: Tensor, cu_seqlens: Tensor, rotary_pos_emb: Tensor) -> Tensor:
        # hidden_states (total_patches, vision_model_hidden_size)
        # cu_seqlens (total_times + 1, )
        # rotary_pos_emb (total_patches, head_dim // 2)
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb) # (total_patches, vision_model_hidden_size)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states)) # (total_patches, vision_model_hidden_size)
        return hidden_states

class Qwen2VisionTransformerPretrainedModelMock(nn.Module):
    def __init__(self, model_path: str, dtype: torch.dtype, device: torch.device):
        super().__init__()
        # 1. config
        config = Qwen2VLVisionConfig.from_pretrained(model_path)
        self.spatial_merge_size = config.spatial_merge_size
        
        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )

        head_dim = config.embed_dim // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [Qwen2VLVisionBlock(config, config._attn_implementation) for _ in range(config.depth)]
        )
        self.merger = PatchMerger(
            dim=config.hidden_size, context_dim=config.embed_dim, spatial_merge_size=config.spatial_merge_size
        )

    def rot_pos_emb(self, grid_thw: Tensor) -> Tensor:
        # grid_thw (n_images, 3)
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w) # (height, width) eg. [[0, 0, 0], [1, 1, 1]] when height=2, width=3
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            ) # (height / spatial_merge_size, spatial_merge_size, width / spatial_merge_size, spatial_merge_size)
            hpos_ids = hpos_ids.permute(0, 2, 1, 3) # (height / spatial_merge_size, width / spatial_merge_size, spatial_merge_size, spatial_merge_size)
            hpos_ids = hpos_ids.flatten() # (n_patches, )

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten() # (n_patches, )
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1)) # (n_patches * t, 2)
        pos_ids = torch.cat(pos_ids, dim=0) # (n_patches all times all images, 2)
        max_grid_size = grid_thw[:, 1:].max().item() # max height and width of all images
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size) # (max_grid_size, head_dim // 4)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1) # (n_patches all times all images, head_dim // 2)
        return rotary_pos_emb
    
    def forward(self, hidden_states: Tensor, grid_thw: Tensor) -> VisionModelOutput:
        # hidden_states (total_patches, 1176)
        hidden_states = self.patch_embed(hidden_states) # (total_patch, vision_model_hidden_size)
        rotary_pos_emb = self.rot_pos_emb(grid_thw) # (total_patch, vision_model_hidden_size)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(dim=0, dtype=torch.int32)
        # after interleave [n_patches1, n_patches1, ... n_patches1, n_patches2, ...]
        # cu_seqlens (total times, )
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0) # insert one 0 before and zero 0 after in last dim
        # cu_seqlens (total times + 1, )

        cu_seqlens = cu_seqlens.to(hidden_states.device)
        for blk in self.blocks:
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)
        # hidden_states (n_patches, vision_model_hidden_size)

        return self.merger(hidden_states) # (n_patches // 4, vision_model_hidden_size)

class Qwen2VisionModel(VisionModel):
    def __init__(self, path: str, dtype: torch.dtype, device: torch.device):
        self.config = Qwen2VLConfig.from_pretrained(path)
        with torch.device(device):
            self.visual = Qwen2VisionTransformerPretrainedModelMock(path, dtype, device)

        load_safetensor(model_with_prefix_list=[(self.visual, 'visual.')], param_with_name_list=[], model_weights_path=path)
        self.visual.to(dtype)
        self.visual.eval()


    def forward(self, pixel_values: list[Tensor], model_params: VisionModelParameters) -> VisionModelOutput:
        # pixel_values a list of (n_patches=(height + 13) // 14 * (width + 13) / 14 , 1176=2*3*14*14)
        pixel_values_list: list[Tensor] = []
        grid_thws: list[Tensor] = []
        for i, pixel_values_per_request in enumerate(pixel_values):
            # pixel_values_per_request (n_patches, 1176) for one image
            pixel_values_list.append(pixel_values_per_request)
            height, width = model_params.original_image_sizes[i]
            height, width = smart_resize(height, width)
            # thws: (1, 3) i.e. [[1, height, width]]
            grid_thws.append(torch.tensor([[1, height // 14, width // 14]], dtype=torch.int64))

        pixel_values = torch.cat(pixel_values, dim=0) # (total_patches, 1176)
        grid_thws: Tensor = torch.cat(grid_thws, dim=0) # (n_image, 3)

        output = VisionModelOutput()
        output.image_features = self.visual.forward(pixel_values, grid_thws) # (total_patches / 4, hidden_size of language model)
        return output

class Qwen2VLSdpaAttention(nn.Module):
    def __init__(self, config: Qwen2VLConfig):
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=True)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(
            rotary_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            inv_freq=compute_default_inv_freq(
                rotary_dim=config.head_dim,
                theta=config.rope_theta
                ),
            interleaved=False
        )

        self.attention = ROPECausalGroupedQueryPageAttention(
            q_proj = self.q_proj, 
            k_proj = self.k_proj, 
            v_proj = self.v_proj, 
            o_proj = self.o_proj, 
            rotary_emb = self.rotary_emb, 
            n_qo_heads = config.num_attention_heads, 
            n_kv_heads = config.num_key_value_heads, 
            head_dim = config.head_dim, 
        )
    
    def forward(self, hidden_states: Tensor, position_ids: Tensor, attention_param: AttentionParameters) -> Tensor:
        return self.attention.forward(hidden_states, position_ids, attention_param)


class Qwen2MLP(nn.Module):
    def __init__(self, config: Qwen2VLConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.activation = Silu()
        self.mlp = GateUpDownMLP(gate_proj=self.gate_proj, up_proj=self.up_proj, down_proj=self.down_proj, activation=self.activation)
    
    def forward(self, h: Tensor) -> Tensor:
        return self.mlp.forward(h)


class Qwen2VLDecoderLayer(nn.Module):
    def __init__(self, config: Qwen2VLConfig, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.self_attn = Qwen2VLSdpaAttention(config)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)

        self.decoder_layer = DecoderLayer(attention=self.self_attn, mlp=self.mlp, norm_1=self.input_layernorm, norm_2=self.post_attention_layernorm, layer_id=layer_id, n_layers= config.num_hidden_layers)
    
    def forward(self, hidden_states: Tensor, position_ids: Tensor, model_params: LanguageModelParameters) -> Tensor:
        return self.decoder_layer.forward(hidden_states, position_ids, model_params)


class Qwen2VLModel(nn.Module):
    """
    Note Qwen2VLForConditionalGeneration has three parts:
    1. visual: Qwen2VisionTransformerPretrainedModel
    2. model: **Qwen2VLModel** (this module)
    3. lm_head: Linear
    """
    def __init__(self, config: Qwen2VLConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size)
        self.layers = nn.ModuleList([Qwen2VLDecoderLayer(config, layer_id) for layer_id in range(config.num_hidden_layers)])
        self.norm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(self, input_ids: Tensor, position_ids: Tensor, model_params: LanguageModelParameters) -> Tensor:
        if input_ids.dtype == torch.int:
            # input_ids is token_ids
            hidden_states = self.embed_tokens(input_ids)
        else:
            # input_ids is input embeds
            hidden_states = input_ids

        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, position_ids, model_params)
        return self.norm(hidden_states)

class Qwen2VLForConditionalGeneration(nn.Module):
    """
    This module mocks Qwen2VL for language generation only.
    """
    def __init__(self, config: Qwen2VLConfig):
        super().__init__()
        self.config = config
        self.config.head_dim = config.hidden_size // config.num_attention_heads
        self.model = Qwen2VLModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def forward(self, input_ids: Tensor, image_features: Optional[Tensor], position_ids: Tensor, model_params: LanguageModelParameters) -> Tensor:
        # input_ids (n_tokens, )
        # input_ids      (n_text_tokens + n_image_tokens) n_text_tokens is number of text tokens, n_image_tokens is number of image tokens
        # image_features (n_image_tokens, hidden_size)
        # position_ids (n_text_tokens + n_image_tokens)
        input_embeds = self.model.embed_tokens(input_ids) # (n_tokens, hidden_size)
        if image_features is not None:
            image_overwrite_mask = input_ids == self.config.image_token_id
            input_embeds[image_overwrite_mask, :] = image_features.view(-1, input_embeds.shape[-1])
        
        hidden_state = self.model(input_embeds, position_ids, model_params) # hidden_state (n_selected_tokens, hidden_size) we discard tokens that do not need to be sampled before entering into the last ffn layer to reduce redundant computation
        logits = self.lm_head(hidden_state) # (n_selected_tokens, hidden_size)
        sample_token_ids = torch.argmax(logits, dim=-1, keepdim=False) # (n_selected_tokens, )
        return sample_token_ids
    
    @classmethod
    def from_safetensor(cls, model_weights_path: str, dtype: torch.dtype, device: torch.device):
        # 1. create model
        config = Qwen2VLConfig.from_pretrained(model_weights_path)
        torch.set_default_dtype(dtype)
        with torch.device(device):
            model = cls(config)
        torch.set_default_dtype(torch.float)

        # 2. load weights
        load_safetensor(model_with_prefix_list=[(model, '')], param_with_name_list=[], model_weights_path=model_weights_path)

        model.to(dtype)
        model.eval()

        return model

class Qwen2VLLanguageModel(LanguageModel):
    def __init__(self, model_path: str, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.model = Qwen2VLForConditionalGeneration.from_safetensor(model_path, dtype, device)
        self.config = self.model.config

    def forward(self, input_ids: Tensor, image_features: Optional[Tensor], position_ids: Tensor, model_params: LanguageModelParameters) -> LanguageModelOutput:
        sample_token_ids = self.model(input_ids, image_features, position_ids, model_params)
        return LanguageModelOutput(sample_token_ids=sample_token_ids)
    

class Qwen2VLTokenizer(Tokenizer):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id
        current_script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(current_script_path)
        template_path = os.path.join(script_dir, "chat_template", "template_qwen_vl_chat.jinja")
        with open(template_path, 'r', encoding='utf-8') as file:
            self.chat_template = Template(file.read())

    def encode(self, prompt: str) -> list[int]:
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        return token_ids

    def decode(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id])

    def apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        prompt = self.chat_template.render(
            messages = messages, 
            bos_token = self.bos_token, 
            eos_token = self.eos_token,
            add_generation_prompt = True, 
        )
        return prompt


class Qwen2VLModelFactory(ModelFactory):
    def __init__(self, config: ModelFactoryConfig, context: ModelFactoryContext):
        self.path = config.path
        self.dtype = str2dtype(config.dtype)
        self.device = str2device(config.device)

    def getVisionModel(self) -> VisionModel:
        return Qwen2VisionModel(self.path, self.dtype, self.device)

    def getLanguageModel(self) -> LanguageModel:
        return Qwen2VLLanguageModel(self.path, self.dtype, self.device)

    def getVisionModelConfig(self) -> VisionModelConfig:
        config_ref = Qwen2VLConfig.from_pretrained(self.path)

        # In qwen2-vl, the prompt <|vision_start|><|image_pad|><|vision_end|>
        # will be encoded into 3 tokens.
        config = VisionModelConfig(
            # represent a image here
            image_token = "<|vision_start|><|image_pad|><|vision_end|>",
            # the <|image_pad|> token id
            image_token_id = config_ref.image_token_id,
            # use calculator to determine the number of image tokens
            image_token_caculator = Qwen2VLImageTokenCaculator(),
        )
        return config

    def getLanguageModelConfig(self) -> LanguageModelConfig:
        config_ref = Qwen2VLConfig.from_pretrained(self.path)
        tokenzier = AutoTokenizer.from_pretrained(self.path)

        n_layers = config_ref.num_hidden_layers
        max_position_embeddings = config_ref.max_position_embeddings
        hidden_size = config_ref.hidden_size
        n_qo_heads = config_ref.num_attention_heads
        n_kv_heads = config_ref.num_key_value_heads
        assert hidden_size % n_qo_heads == 0
        head_dim = hidden_size // n_qo_heads

        config = LanguageModelConfig(
            n_layers=n_layers,
            max_position_embeddings=max_position_embeddings, 
            n_qo_heads=n_qo_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            eos_token_id = tokenzier.eos_token_id, 
        )
        return config

    def getProcessor(self) -> ImageProcessor:
        return TransformersAutoProcessorAdapter(self.path)

    def getTokenizer(self) -> Tokenizer:
        return Qwen2VLTokenizer(self.path)

    def getModelProfiler(self) -> ModelProfiler:
        return VisionLanguageModelProfiler(self.path, vision_model_prefixes=['visual.'], language_model_prefixes=['model.', 'lm_head.'])