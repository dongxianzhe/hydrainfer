import torch
from torch import Tensor
from dataclasses import dataclass
from dxz.layer.causal_attention import AttentionParameters


@dataclass
class VisionModelParameters:
    return_last_layer_attention: bool = False


@dataclass
class VisionModelOutput:
    image_features: Tensor = None
    scores: Tensor = None


@dataclass
class LanguageModelParameters:
    attention_params: list[AttentionParameters]
    all_sequences_decode: bool
    selected_token_ids: list[int]


@dataclass
class LanguageModelOutput:
    sample_token_ids: Tensor = None