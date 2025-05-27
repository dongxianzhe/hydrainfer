import torch
from torch import nn
from torch import Tensor
from transformers import GPT2Config
from hydrainfer.model.model_factory import ModelFactory, VisionModel, VisionModelConfig, LanguageModel, LanguageModelConfig, ModelFactoryConfig, ModelFactoryContext
from transformers import AutoProcessor, AutoTokenizer
from hydrainfer.model.parameters import LanguageModelParameters, LanguageModelOutput, AttentionParameters
from hydrainfer.model.downloader import download_hf_model
from hydrainfer.layer.activation import NewGELUActivation
from hydrainfer.layer.causal_attention import CausalGroupedQueryPageAttention, CausalGroupedQueryPageAttentionConfig
from hydrainfer.utils.torch_utils import str2dtype, str2device


class GPT2Attention(nn.Module):
    def __init__(self, config: GPT2Config):
        super(GPT2Attention, self).__init__()
        self.config = config
        self.c_attn = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=True)
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.atten = CausalGroupedQueryPageAttention(CausalGroupedQueryPageAttentionConfig(
            n_qo_heads=config.n_head,
            n_kv_heads=config.n_head,
            head_dim=config.hidden_size // config.n_head
        ))
    
    def forward(self, hidden_states: Tensor, params: AttentionParameters) -> Tensor:
        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        attn_output = self.atten(q, k, v, params).o
        attn_output = self.c_proj(attn_output)
        return attn_output


class GPT2MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super(GPT2MLP, self).__init__()
        assert config.n_inner is None, "only support intermediate = 4 * hidden_size now"
        intermediate_size = config.hidden_size * 4
        self.c_fc = nn.Linear(config.hidden_size, intermediate_size, bias=True)
        self.c_proj = nn.Linear(intermediate_size, config.hidden_size, bias=True)
        self.act = NewGELUActivation()
    
    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states


class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config, layer_id: int):
        super(GPT2Block, self).__init__()
        self.layer_id = layer_id
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config=config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)

    def forward(self, hidden_states: Tensor, params: LanguageModelParameters) -> Tensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states, params.attention_params[self.layer_id])
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states
        return hidden_states


class GPT2Model(nn.Module):
    def __init__(self, config: GPT2Config):
        super(GPT2Model, self).__init__()
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.h = nn.ModuleList([GPT2Block(config, layer_id) for layer_id in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
    
    def forward(self, input_ids: Tensor, position_ids: Tensor, params: LanguageModelParameters) -> dict[str, Tensor]:
        input_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = input_embeds + position_embeds

        for layer in self.h:
            hidden_states = layer(hidden_states, params)
        hidden_states = self.ln_f(hidden_states)
        return hidden_states
    

class GPT2LMHeadModel(nn.Module):
    def __init__(self, config: GPT2Config):
        super(GPT2LMHeadModel, self).__init__()
        self.config = config
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
    
    def forward(self, input_ids: Tensor, position_ids: Tensor, params: LanguageModelParameters) -> LanguageModelOutput:
        hidden_state = self.transformer(input_ids, position_ids, params)
        logits = self.lm_head(hidden_state)
        logits = logits[params.selected_token_ids, :]
        sample_token_ids = torch.argmax(logits, dim=-1, keepdim=False)
        return LanguageModelOutput(sample_token_ids=sample_token_ids)
    
    @classmethod
    def from_pretrained(cls, model_weights_path: str, dtype: torch.dtype, device: torch.device):
        from transformers import GPT2LMHeadModel as GPT2LMHeadModelRef

        with torch.device(device):
            model_ref = GPT2LMHeadModelRef.from_pretrained(model_weights_path)
            model = cls(model_ref.config)

        state_dict_ref = model_ref.state_dict()
        for name, weight in model.state_dict().items():
            if any(needle in name for needle in ["c_attn.weight", "c_proj.weight", "c_fc.weight"]):
                weight.data.copy_(state_dict_ref[name].t())
            else:
                weight.data.copy_(state_dict_ref[name])

        model.to(dtype)
        model.eval()

        return model


class GPT2VisionModel(VisionModel):
    pass


class GPT2LanguageModel(LanguageModel):
    def __init__(self, model_weights_path: str, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_weights_path, dtype, device)
        self.config = self.model.config
    
    def forward(self, input_ids, image_features, position_ids, model_params):
        return self.model(input_ids, position_ids, model_params)


class GPT2ModelFactory(ModelFactory):
    def __init__(self, config: ModelFactoryConfig, context: ModelFactoryContext):
        if config.path is None:
            self.path = download_hf_model(repo_id=config.name)
        else:
            self.path = config.path
        self.dtype = str2dtype(config.dtype)
        self.device = str2device(config.device)


    def getVisionModel(self) -> VisionModel:
        return GPT2VisionModel()

    def getLanguageModel(self) -> LanguageModel:
        return GPT2LanguageModel(self.path, self.dtype, self.device)

    def getVisionModelConfig(self)-> VisionModelConfig:
        config = VisionModelConfig(image_token_id = -1, num_image_tokens = -1)
        return config

    def getLanguageModelConfig(self,) -> LanguageModelConfig:
        from transformers import GPT2LMHeadModel as GPT2LMHeadModelRef
        model_ref = GPT2LMHeadModelRef.from_pretrained(self.path)
        config_ref = model_ref.config
        config = LanguageModelConfig(
            n_layers = config_ref.n_layer, 
            max_position_embeddings = config_ref.n_positions, 
            n_qo_heads = config_ref.n_head, 
            n_kv_heads = config_ref.n_head, 
            head_dim = config_ref.n_embd // config_ref.n_head, 
        )
        return config

    def getProcessor(self) -> AutoProcessor:
        return None

    def getTokenizer(self) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(self.path)