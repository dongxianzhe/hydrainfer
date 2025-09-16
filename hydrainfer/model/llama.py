import os
import torch
from torch import nn, Tensor
from transformers import LlamaConfig, AutoProcessor, AutoTokenizer
from typing import Optional
import safetensors.torch
from hydrainfer.layer.rotary_embedding import RotaryEmbedding, compute_default_inv_freq
from hydrainfer.model.downloader import download_hf_model
from hydrainfer.model.parameters import LanguageModelParameters, AttentionParameters
from hydrainfer.model.parameters import VisionModelParameters, VisionModelOutput, LanguageModelParameters, LanguageModelOutput
from hydrainfer.model.model_factory import VisionModel, VisionModelConfig, LanguageModel, LanguageModelConfig, ModelFactory, ModelFactoryConfig, ModelFactoryContext
from hydrainfer.layer.causal_attention import CausalGroupedQueryPageAttention, CausalGroupedQueryPageAttentionConfig
from hydrainfer.layer.norm import RMSNorm
from hydrainfer.layer.activation import silu
from hydrainfer.utils.torch_utils import str2dtype, str2device
from hydrainfer.utils.logger import getLogger
from hydrainfer.model.model_forward import ROPECausalGroupedQueryPageAttention, GateUpDownMLP, DecoderLayer
logger = getLogger(__name__)

class LlamaSdpaAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
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

        self.attention = ROPECausalGroupedQueryPageAttention(q_proj = self.q_proj, k_proj = self.k_proj, v_proj = self.v_proj, o_proj = self.o_proj, rotary_emb = self.rotary_emb, n_qo_heads = config.num_attention_heads, n_kv_heads = config.num_key_value_heads, head_dim = config.head_dim )
    
    def forward(self, hidden_states: Tensor, position_ids: Tensor, attention_param: AttentionParameters) -> Tensor:
        # hidden_states (n_tokens, hidden_size) position_ids (n_tokens, )
        return self.attention.forward(hidden_states, position_ids, attention_param)

class LlamaMLP(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

        self.mlp = GateUpDownMLP(gate_proj = self.gate_proj, up_proj = self.up_proj, down_proj = self.down_proj, activation = silu)
    
    def forward(self, hidden_states: Tensor) -> Tensor:
        # hidden_states (n_tokens, hidden_size)
        return self.mlp.forward(hidden_states)


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.self_attn = LlamaSdpaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)

        self.decoder_layer = DecoderLayer(attention = self.self_attn, mlp = self.mlp, norm_1 = self.input_layernorm, norm_2 = self.post_attention_layernorm, layer_id = self.layer_id, n_layers = config.num_hidden_layers)
        
    
    def forward(self, hidden_states: Tensor, position_ids: Tensor, model_params: LanguageModelParameters) -> Tensor:
        return self.decoder_layer.forward(hidden_states, position_ids, model_params)

class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, layer_id) for layer_id in range(config.num_hidden_layers)])
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

class LlamaForCausalLM(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def forward(self, input_ids_or_input_embeds: Tensor, position_ids: Tensor, model_params: LanguageModelParameters) -> Tensor:
        # input_ids (n_tokens, ) or input embeds (n_tokens, hidden_size)
        hidden_state = self.model(input_ids_or_input_embeds, position_ids, model_params) # hidden_state (n_selected_tokens, hidden_size) we discard tokens that do not need to be sampled before entering into the last ffn layer to reduce redundant computation
        logits = self.lm_head(hidden_state) # (n_selected_tokens, hidden_size)
        sample_token_ids = torch.argmax(logits, dim=-1, keepdim=False) # (n_selected_tokens, )
        return sample_token_ids
    
    @classmethod
    def from_safetensor(cls, model_weights_path: str, dtype: torch.dtype, device: torch.device):
        # 1. create model
        config = LlamaConfig.from_pretrained(model_weights_path)
        torch.set_default_dtype(dtype)
        with torch.device(device):
            model = cls(config)
        torch.set_default_dtype(torch.float)

        # 2. load weights
        state_dict = model.state_dict()
        loaded_set = set()
        for entry in os.scandir(model_weights_path):
            if entry.is_file() and os.path.splitext(entry.name)[1] == '.safetensors':
                logger.info(f'load safetensor from {entry.path}')
                for name, weight in safetensors.torch.load_file(entry.path).items():
                    if name.endswith('.self_attn.rotary_emb.inv_freq'):
                        continue
                    state_dict[name].data.copy_(weight)
                    loaded_set.add(name)
        model.load_state_dict(state_dict)
        model.to(dtype)
        model.eval()

        # 3. verify
        assert len(loaded_set) == len(state_dict)

        return model


class LlamaVisionModel(VisionModel):
    pass


class LlamaLanguageModel(LanguageModel):
    def __init__(self, model_path: str, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.model = LlamaForCausalLM.from_safetensor(model_path, dtype, device)
        self.config = self.model.config

    def forward(self, input_ids: Tensor, image_features: Optional[Tensor], position_ids: Tensor, model_params: LanguageModelParameters) -> LanguageModelOutput:
        sample_token_ids = self.model(input_ids, position_ids, model_params)
        return LanguageModelOutput(sample_token_ids=sample_token_ids)
    

class LlamaModelFactory(ModelFactory):
    def __init__(self, config: ModelFactoryConfig, context: ModelFactoryContext):
        self.name = config.name
        if config.path is None:
            self.path = download_hf_model(repo_id=config.name)
        else:
            self.path = config.path
        self.dtype = str2dtype(config.dtype)
        self.device = str2device(config.device)

    def getVisionModel(self) -> VisionModel:
        return LlamaVisionModel()

    def getLanguageModel(self) -> LanguageModel:
        model = LlamaLanguageModel(self.path, self.dtype, self.device)
        return model

    def getVisionModelConfig(self) -> VisionModelConfig:
        return VisionModelConfig(image_token_id = -1, num_image_tokens = -1)

    def getLanguageModelConfig(self) -> LanguageModelConfig:
        config_ref = LlamaConfig.from_pretrained(self.path)
        config = LanguageModelConfig(
            n_layers = config_ref.num_hidden_layers, 
            max_position_embeddings = config_ref.max_position_embeddings, 
            n_qo_heads = config_ref.num_attention_heads, 
            n_kv_heads = config_ref.num_key_value_heads, 
            head_dim = config_ref.head_dim
        )
        return config

    def getProcessor(self) -> AutoProcessor:
        return None

    def getTokenizer(self) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(self.path)