from PIL import Image
from typing import Optional, Literal
from dataclasses import dataclass, field


@dataclass
class SamplingParameters:
    max_tokens: int = 50


@dataclass
class TokenParameters:
    kv_cache_eviction_policy: Literal[None, 'random', 'streamingllm'] = None
    window_size: int = 28
    attention_sink_size: int = 4 
    token_pruning_policy: Literal[None, 'focal'] = None
    n_embed_output_tokens: int = 64


@dataclass
class Request:
    request_id: int = 0
    prompt: str = ""
    image: Optional[Image.Image] = None
    image_base64: str = ""
    sampling_params: SamplingParameters = field(default_factory=SamplingParameters)
    token_params: TokenParameters = field(default_factory=TokenParameters)