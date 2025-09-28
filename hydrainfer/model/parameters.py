import torch
from torch import Tensor
from dataclasses import dataclass
from hydrainfer.layer.causal_attention import AttentionParameters
from typing import Optional


@dataclass
class VisionModelParameters:
    return_last_layer_attention: bool = False
    original_image_sizes: Optional[list[tuple[int, int]]] = None # used in llavanext image unpadding


@dataclass
class VisionModelOutput:
    image_features: Tensor = None
    scores: Tensor = None


@dataclass
class LanguageModelParameters:
    input_ids_or_input_embeds: Tensor
    position_ids: Tensor
    image_features: Optional[Tensor]
    image_overwrite_mask: Optional[Tensor]
    attention_params: list[AttentionParameters]
    all_sequences_decode: bool
    selected_token_ids: list[int]


@dataclass
class LanguageModelOutput:
    sample_token_ids: Tensor = None