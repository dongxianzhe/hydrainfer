import math
import torch
from torch import nn
from torch import Tensor

class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

class GPT2Attention(nn.Module):
    def __init__(self, config, layer_idx=None):
        self.c_attn = nn.Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = nn.Conv1D(self.embed_dim, self.embed_dim)
        self.atten = nn.MultiheadAttention()

class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config):
        embed_dim = config.hidden_size
        self.c_fc = nn.Conv1D(intermediate_size, embed_dim)
        self.c_proj = nn.Conv1D(embed_dim, intermediate_size)
        self.act = NewGELUActivation()

class GPT2Block(nn.Module):
    def __init__(self, config, layer_idx):
        super(GPT2Block, self).__init__()
        hidden_size = config.hidden_size
        inner_dim = 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config=config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(inner_dim, config)

class GPT2Model(nn.Module):
    def __init__(self, config):
        super(GPT2Model, self).__init__()
        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
