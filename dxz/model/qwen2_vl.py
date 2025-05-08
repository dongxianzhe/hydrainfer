import os
import torch
from torch import nn, Tensor
from transformers import Qwen2VLConfig, AutoProcessor, AutoTokenizer
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel
from typing import Optional
import safetensors.torch
from dxz.layer.rotary_embedding import RotaryEmbedding, compute_default_inv_freq
from dxz.model.downloader import download_hf_model
from dxz.model.parameters import AttentionParameters, LanguageModelParameters, LanguageModelOutput, VisionModelParameters, VisionModelOutput
from dxz.model.model_factory import VisionModel, VisionModelConfig, LanguageModel, LanguageModelConfig, ModelFactory, ModelFactoryConfig, ModelFactoryContext, ImageTokenCaculator
from dxz.layer.causal_attention import CausalGroupedQueryPageAttention, CausalGroupedQueryPageAttentionConfig
from dxz.layer.norm import rmsnorm
from dxz.layer.activation import silu
from dxz.utils.torch_utils import str2dtype, str2device

class Qwen2VLImageTokenCaculator(ImageTokenCaculator):
    def get_num_image_tokens(self, image_size: tuple[int, int]) -> int:
        """
        image_size (height, width)
        """
        height, width = image_size
        height, width = smart_resize(height,width)
        return (height // 14) * (width // 14) // 4

class Qwen2VisionTransformerPretrainedModelMock(nn.Module):
    def __init__(self, model_path: str, dtype: torch.dtype, device: torch.device):
        super().__init__()
        # 1. config
        config = Qwen2VLConfig.from_pretrained(model_path)
        # 2. create model
        torch.set_default_dtype(dtype)
        with torch.device(device):
            self.vision = Qwen2VisionTransformerPretrainedModel(config.vision_config)
        torch.set_default_dtype(torch.float)
        # 3. load vision state dict
        state_dict = self.vision.state_dict()
        loaded_set = set() # used to verify all weight are loaded
        for entry in os.scandir(model_path):
            if entry.is_file() and os.path.splitext(entry.name)[1] == '.safetensors':
                print(f'load safetensor from {entry.path}')
                for name, weight in safetensors.torch.load_file(entry.path).items():
                    if name.startswith('visual.'):
                        state_dict[name.removeprefix('visual.')].copy_(weight)
                        loaded_set.add(name)
        self.vision.load_state_dict(state_dict)
        self.vision.to(dtype)
        self.vision.eval()
        # 4. verify
        assert len(state_dict) == len(loaded_set), f'{len(state_dict)} {len(loaded_set)}'

    
    def forward(self, pixel_values: Tensor, grid_thws: Tensor) -> VisionModelOutput:
        assert pixel_values.dim() == 2, f'pixel value shape should be 4 dim but got {pixel_values.shape}'
        output = VisionModelOutput()
        output.image_features = self.vision(pixel_values, grid_thws) # (n_patches, hidden_size)
        return output

class Qwen2VisionModel(VisionModel):
    def __init__(self, path: str, dtype: torch.dtype, device: torch.device):
        self.config = Qwen2VLConfig.from_pretrained(path)
        self.visual = Qwen2VisionTransformerPretrainedModelMock(path, dtype, device)

    def forward(self, pixel_values: list[Tensor], model_params: VisionModelParameters) -> VisionModelOutput:
        pixel_values_list: list[Tensor] = []
        grid_thws: list[Tensor] = []
        for i, pixel_values_per_request in enumerate(pixel_values):
            # pixel_values_per_request (n_patches, 1176) for one image
            pixel_values_list.append(pixel_values_per_request)
            height, width = model_params.original_image_sizes[i]
            height, width = smart_resize(height, width)
            # thws: (1, 3) i.e. [[1, height, width]]
            grid_thws.append(torch.tensor([1, height // 14, width // 14], dtype=torch.int64))

        pixel_values = torch.cat(pixel_values, dim=0)
        grid_thws = torch.cat(grid_thws, dim=0)

        output = self.visual.forward(pixel_values, grid_thws)
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
        self.attention = CausalGroupedQueryPageAttention(CausalGroupedQueryPageAttentionConfig(
            n_qo_heads=config.num_attention_heads,
            n_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim
        ))
    
    def forward(self, hidden_states: Tensor, position_ids: Tensor, attention_param: AttentionParameters) -> Tensor:
        # hidden_states (n_tokens, hidden_size)
        # position_ids (n_tokens, )
        query = self.q_proj(hidden_states) # (n_tokesn, hidden_size)
        key   = self.k_proj(hidden_states) # (n_tokesn, hidden_size)
        value = self.v_proj(hidden_states) # (n_tokesn, hidden_size)
        query = query.view(-1, self.config.num_attention_heads, self.config.head_dim) # (n_tokens, n_qo_heads, head_size)
        key   = key  .view(-1, self.config.num_key_value_heads, self.config.head_dim) # (n_tokens, n_kv_heads, head_size)
        value = value.view(-1, self.config.num_key_value_heads, self.config.head_dim) # (n_tokens, n_kv_heads, head_size)
        query, key = self.rotary_emb(query, key, position_ids) # query (n_tokens, n_qo_heads, head_size) key (n_tokens, n_kv_heads, head_size) note that rotary_emb is inplace operation
        hidden_states = self.attention(query, key, value, attention_param).o # (n_tokens, hidden_size)
        return self.o_proj(hidden_states) # (n_tokens, hidden_size)

class Qwen2MLP(nn.Module):
    def __init__(self, config: Qwen2VLConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
    
    def forward(self, hidden_states: Tensor) -> Tensor:
        # hidden_states (n_tokens, hidden_size)
        down_proj = self.down_proj(silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        return down_proj

class Qwen2RMSNorm(nn.Module):
    def __init__(self, config: Qwen2VLConfig):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_size))
        self.variance_epsilon = config.rms_norm_eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        output = rmsnorm(hidden_states, self.weight, self.variance_epsilon)
        return output

class Qwen2VLDecoderLayer(nn.Module):
    def __init__(self, config: Qwen2VLConfig, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.n_layers = config.num_hidden_layers
        self.self_attn = Qwen2VLSdpaAttention(config)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config)
        self.post_attention_layernorm = Qwen2RMSNorm(config)
    
    def forward(self, hidden_states: Tensor, position_ids: Tensor, model_params: LanguageModelParameters) -> Tensor:
        residual = hidden_states # (n_tokens, hidden_size)
        hidden_states = self.input_layernorm(hidden_states) # (n_tokens, hidden_size)
        hidden_states = self.self_attn(hidden_states, position_ids, model_params.attention_params[self.layer_id]) # (n_tokens, hidden_size)
        hidden_states = residual + hidden_states # (n_tokens, hidden_size)

        # if it is last layer we discared tokens which is not sampled to reduce redundent computation in the last ffn layer
        if not model_params.all_sequences_decode and self.layer_id == self.n_layers - 1:
            hidden_states = hidden_states[model_params.selected_token_ids, :]

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

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
        self.norm = Qwen2RMSNorm(config)
    
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
    
    def forward(self, input_ids_or_input_embeds: Tensor, image_features: Optional[Tensor], position_ids: Tensor, model_params: LanguageModelParameters) -> Tensor:
        # FIXME: add RoPE index
        # input_ids (n_tokens, ) or input embeds (n_tokens, hidden_size)
        # input_ids      (n_text_tokens + n_image_tokens) n_text_tokens is number of text tokens, n_image_tokens is number of image tokens
        # image_features (n_image_tokens, hidden_size)
        # position_ids (n_text_tokens + n_image_tokens)
        hidden_state = self.model(input_ids_or_input_embeds, position_ids, model_params) # hidden_state (n_selected_tokens, hidden_size) we discard tokens that do not need to be sampled before entering into the last ffn layer to reduce redundant computation
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
        state_dict = model.state_dict()
        loaded_set = set()
        for entry in os.scandir(model_weights_path):
            if entry.is_file() and os.path.splitext(entry.name)[1] == '.safetensors':
                print(f'load safetensor from {entry.path}')
                for name, weight in safetensors.torch.load_file(entry.path).items():
                    if name.startswith('visual'):
                        continue
                    state_dict[name].data.copy_(weight)
                    loaded_set.add(name)
        model.load_state_dict(state_dict)
        model.to(dtype)
        model.eval()

        # 3. verify
        assert len(loaded_set) == len(state_dict)

        return model

class Qwen2VLLanguageModel(LanguageModel):
    def __init__(self, model_path: str, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.model = Qwen2VLForConditionalGeneration.from_safetensor(model_path, dtype, device)
        self.config = self.model.config

    def forward(self, input_ids: Tensor, image_features: Optional[Tensor], position_ids: Tensor, model_params: LanguageModelParameters) -> LanguageModelOutput:
        sample_token_ids = self.model(input_ids, image_features, position_ids, model_params)
        return LanguageModelOutput(sample_token_ids=sample_token_ids)
    

class Qwen2VLModelFactory(ModelFactory):
    def __init__(self, config: ModelFactoryConfig, context: ModelFactoryContext):
        self.name = config.name
        if config.path is None:
            self.path = download_hf_model(repo_id=config.name)
        else:
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
            # the <|image_pad|> token id
            image_token_id = config_ref.image_token_id,
            # we don't use num_image_tokens, because it is not sure
            num_image_tokens = 334,
            # use calculator to determine the number of image tokens
            image_token_caculator = Qwen2VLImageTokenCaculator(),
        )
        return config

    def getLanguageModelConfig(self) -> LanguageModelConfig:
        config_ref = Qwen2VLConfig.from_pretrained(self.path)

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
        )
        return config

    def getProcessor(self) -> AutoProcessor:
        return AutoProcessor.from_pretrained(self.path)

    def getTokenizer(self) -> AutoTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(self.path)
        orginal_tokenizer_encode = tokenizer.encode
        def encode(
            self, 
            text: str,
            **kwargs
        ) -> Tensor:
            # replace all <image> with <|vision_start|><|image_pad|><|vision_end|>
            text = text.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
            return orginal_tokenizer_encode(text, **kwargs)
        tokenizer.encode = encode.__get__(tokenizer)
        return tokenizer