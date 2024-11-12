import torch
from torch import nn, Tensor
from dxz.memory.kv_cache import KVCache
from dxz.model.parameters import InputParameters
from transformers import GPT2Config

class FakeLLMModel(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.fake_params = nn.Parameter(torch.ones(vocab_size))
        self.vocab_size = vocab_size

    def forward(self, input_ids: Tensor, position_ids: Tensor, kv_caches: list[KVCache], input_parmas: InputParameters) -> Tensor:
        n_tokens = input_ids.shape[0]
        return torch.randn(n_tokens, self.vocab_size, dtype=self.fake_params.dtype, device=self.fake_params.device)

class FakeMLLMModel(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.fake_params = nn.Parameter(torch.ones(vocab_size))
        self.vocab_size = vocab_size

    def forward(self, input_ids: Tensor, pixel_values: Tensor, position_ids: Tensor, kv_caches: list[KVCache], input_parmas: InputParameters) -> Tensor:
        n_tokens = input_ids.shape[0]
        return torch.randn(n_tokens, self.vocab_size, dtype=self.fake_params.dtype, device=self.fake_params.device)