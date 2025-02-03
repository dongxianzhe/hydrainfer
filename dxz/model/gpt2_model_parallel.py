import torch
from torch import nn, Tensor
from transformers import GPT2Config, AutoProcessor, AutoTokenizer
from transformers import GPT2LMHeadModel as GPT2LMHeadModelRef
from dataclasses import dataclass
from dxz.layer.linear import ColumnParallelLinear, RowParallelLinear
from dxz.layer.activation import NewGELUActivation
from dxz.layer.causal_attention import CausalGroupedQueryPageAttention, CausalGroupedQueryPageAttentionConfig, AttentionParameters
from dxz.model_parallel.process_group import ParallelConfig, ProcessGroup, init_global_process_group
from dxz.model.parameters import LanguageModelParameters, LanguageModelOutput
from dxz.model.downloader import download_hf_model
from dxz.model.model_factory import VisionModel, LanguageModel, ModelFactory, VisionModelConfig, LanguageModelConfig, ModelFactoryConfig, ModelFactoryContext


@dataclass
class ModelContext:
    process_group: ProcessGroup


class GPT2MLP(nn.Module):
    def __init__(self, config: GPT2Config, context: ModelContext):
        super(GPT2MLP, self).__init__()
        intermediate_size = config.hidden_size * 4
        self.c_fc = ColumnParallelLinear(
            in_features = config.hidden_size, 
            out_features = intermediate_size, 
            bias = True, 
            gather_output = False, 
            process_group = context.process_group, 
        )
        self.c_proj = RowParallelLinear(
            in_features = intermediate_size,
            out_features = config.hidden_size, 
            bias = True, 
            input_is_parallelized = False, 
            process_group = context.process_group, 
        )
        self.act = NewGELUActivation()

    def forward(self, h: Tensor) -> Tensor:
        h = self.c_fc(h)
        h = self.act(h)
        h = self.c_proj(h)
        return h

    def load_state_dict(self, state_dict: dict[str, Tensor]):
        self.c_fc.load_state_dict({k.removeprefix('c_fc.'): v for k, v in state_dict.items() if k.startswith('c_fc.')})
        self.c_proj.load_state_dict({k.removeprefix('c_proj.'): v for k, v in state_dict.items() if k.startswith('c_proj.')})


class GPT2Attention(nn.Module):
    def __init__(self, config: GPT2Config, context: ModelContext):
        super(GPT2Attention, self).__init__()
        self.config = config
        self.q_proj = ColumnParallelLinear(
            in_features = config.hidden_size, 
            out_features = config.hidden_size, 
            bias = True, 
            gather_output = False, 
            process_group = context.process_group,
        )
        self.k_proj = ColumnParallelLinear(
            in_features = config.hidden_size, 
            out_features = config.hidden_size, 
            bias = True, 
            gather_output = False, 
            process_group = context.process_group,
        )
        self.v_proj = ColumnParallelLinear(
            in_features = config.hidden_size, 
            out_features = config.hidden_size, 
            bias = True, 
            gather_output = False, 
            process_group = context.process_group,
        )
        n_local_heads = config.n_head // context.process_group.world_size
        self.atten = CausalGroupedQueryPageAttention(CausalGroupedQueryPageAttentionConfig(
            n_qo_heads=n_local_heads,
            n_kv_heads=n_local_heads,
            head_dim=config.hidden_size // config.n_head
        ))
        self.o_proj = RowParallelLinear(
            in_features = config.hidden_size, 
            out_features = config.hidden_size, 
            bias = True, 
            input_is_parallelized = False, 
            process_group = context.process_group,
        )

    def forward(self, h: Tensor, params: AttentionParameters) -> Tensor:
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        o = self.atten(q, k, v, params).o
        o = self.o_proj(o)
        return o

    def load_state_dict(self, state_dict: dict[str, Tensor]):
        weights = torch.chunk(state_dict['c_attn.weight'], chunks=3, dim=0)
        biases = torch.chunk(state_dict['c_attn.bias'], chunks=3, dim=0)
        self.q_proj.load_state_dict({'weight':weights[0], 'bias':biases[0]})
        self.k_proj.load_state_dict({'weight':weights[1], 'bias':biases[1]})
        self.v_proj.load_state_dict({'weight':weights[2], 'bias':biases[2]})
        self.o_proj.load_state_dict({'weight': state_dict['c_proj.weight'], 'bias':state_dict['c_proj.bias']})


class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config, context: ModelContext, layer_id: int):
        super().__init__()
        self.layer_id = layer_id 
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config, context)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config, context)

    def forward(self, h: Tensor, params: LanguageModelParameters) -> Tensor:
        r = h
        h = self.ln_1(h)
        h = self.attn(h, params.attention_params[self.layer_id])
        h = r + h
        r = h
        h = self.ln_2(h)
        h = self.mlp(h)
        h = r + h
        return h

    def load_state_dict(self, state_dict: dict[str, Tensor]):
        self.ln_1.load_state_dict({k.removeprefix('ln_1.'): v for k, v in state_dict.items() if k.startswith('ln_1.')})
        self.ln_2.load_state_dict({k.removeprefix('ln_2.'): v for k, v in state_dict.items() if k.startswith('ln_2.')})
        self.attn.load_state_dict({k.removeprefix('attn.'): v for k, v in state_dict.items() if k.startswith('attn.')})
        self.mlp.load_state_dict({k.removeprefix('mlp.'): v for k, v in state_dict.items() if k.startswith('mlp.')})


class GPT2Model(nn.Module):
    def __init__(self, config: GPT2Config, context: ModelContext):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.h = nn.ModuleList([GPT2Block(config, context, layer_id) for layer_id in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(self, input_ids: Tensor, position_ids: Tensor, params: LanguageModelParameters):
        input_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = input_embeds + position_embeds
        for layer in self.h:
            hidden_states = layer(hidden_states, params)
        hidden_states = self.ln_f(hidden_states)
        return hidden_states

    def load_state_dict(self, state_dict: dict[str, Tensor]):
        self.wte.load_state_dict({k.removeprefix('wte.'): v for k, v in state_dict.items() if k.startswith('wte.')})
        self.wpe.load_state_dict({k.removeprefix('wpe.'): v for k, v in state_dict.items() if k.startswith('wpe.')})
        for layer_id, layer in enumerate(self.h):
            layer.load_state_dict({k.removeprefix(f'h.{layer_id}.'): v for k, v in state_dict.items() if k.startswith(f'h.{layer_id}.')})
        self.ln_f.load_state_dict({k.removeprefix('ln_f.'): v for k, v in state_dict.items() if k.startswith('ln_f.')})


class GPT2LMHeadModel(nn.Module):
    def __init__(self, config: GPT2Config, context: ModelContext):
        super().__init__()
        self.config = config
        self.transformer = GPT2Model(config, context)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: Tensor, position_ids: Tensor, params: LanguageModelParameters) -> LanguageModelOutput:
        hidden_states = self.transformer(input_ids, position_ids, params)
        hidden_states = hidden_states[params.selected_token_ids, :] 
        logits = self.lm_head(hidden_states)
        sample_token_ids = torch.argmax(logits, dim=-1, keepdim= False)
        return LanguageModelOutput(sample_token_ids = sample_token_ids)

    def load_state_dict(self, state_dict: dict[str, Tensor]):
        self.transformer.load_state_dict({k.removeprefix('transformer.'): v for k, v in state_dict.items() if k.startswith('transformer.')})
        self.lm_head.load_state_dict({k.removeprefix('lm_head.'): v for k, v in state_dict.items() if k.startswith('lm_head.')})


class GPT2VisionModel(VisionModel):
    pass


class GPT2LanguageModel(LanguageModel):
    def __init__(self, model_weights_path: str, dtype: torch.dtype, device: torch.device, process_group: ProcessGroup):
        super().__init__()
        # create model
        self.config = GPT2Config.from_pretrained(model_weights_path)
        context = ModelContext(process_group=process_group)
        prev_dtype = torch.get_default_dtype()
        prev_device = torch.get_default_device()
        torch.set_default_dtype(dtype)
        torch.set_default_device(device)
        self.model = GPT2LMHeadModel(self.config, context)
        torch.set_default_dtype(prev_dtype)
        torch.set_default_device(prev_device)

        # load state weights
        model_ref = GPT2LMHeadModelRef.from_pretrained(model_weights_path)
        state_dict_ref = model_ref.state_dict()
        state_dict = {}
        for name, weight in state_dict_ref.items():
            if any(needle in name for needle in ["c_attn.weight", "c_proj.weight", "c_fc.weight"]):
                state_dict[name] = weight.t()
            else:
                state_dict[name] = weight
        self.model.load_state_dict(state_dict)

    
    def forward(self, input_ids, image_features, position_ids, model_params):
        return self.model(input_ids, position_ids, model_params)


class GPT2ModelFactory(ModelFactory):
    def __init__(self, config: ModelFactoryConfig, context: ModelFactoryContext):
        if config.model_path is None:
            self.model_path = download_hf_model(repo_id=config.model_name)
        else:
            self.model_path = config.model_path
        self.dtype = config.dtype
        self.device = config.device
        self.context = context

    def getVisionModel(self) -> VisionModel:
        return GPT2VisionModel()

    def getLanguageModel(self) -> LanguageModel:
        model = GPT2LanguageModel(self.model_path, self.dtype, self.device, process_group=self.context.process_group)
        return model

    def getVisionModelConfig(self) -> VisionModelConfig:
        return VisionModelConfig(image_token_id = -1, num_image_tokens = -1)

    def getLanguageModelConfig(self) -> LanguageModelConfig:
        config_ref = GPT2Config.from_pretrained(self.model_path)
        return LanguageModelConfig(
            n_layers = config_ref.n_layer, 
            max_position_embeddings = config_ref.n_positions, 
            n_qo_heads = config_ref.n_head, 
            n_kv_heads = config_ref.n_head, 
            head_dim = config_ref.n_embd // config_ref.n_head, 
        )

    def getProcessor(self) -> AutoProcessor:
        return None

    def getTokenizer(self) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(self.model_path)