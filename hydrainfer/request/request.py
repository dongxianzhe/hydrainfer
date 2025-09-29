from PIL import Image
from typing import Optional, Literal, Union
from dataclasses import dataclass, field


@dataclass
class SamplingParameters:
    max_tokens: int = 50
    eos_token_ids: list[int] = field(default_factory=list)


@dataclass
class TokenParameters:
    kv_cache_eviction_policy: Literal[None, 'random', 'streamingllm'] = None
    window_size: int = 28
    attention_sink_size: int = 4 
    token_pruning_policy: Literal[None, 'focal'] = None
    n_embed_output_tokens: int = 64


@dataclass
class Request:
    request_id: Union[int, str] = 0
    prompt: str = ""
    image: Optional[Image.Image] = None
    image_base64: Optional[str] = None
    sampling_params: SamplingParameters = field(default_factory=SamplingParameters)
    token_params: TokenParameters = field(default_factory=TokenParameters)


@dataclass
class RequestMetaData:
    """ 
    n_prompt_tokens = n_text_tokens + n_image_tokens
    n_image_tokens = n_image_tokens * n_image_tokens_per_image
    """
    n_images: int
    n_prompt_tokens: int
    n_text_tokens: int
    n_image_tokens: int