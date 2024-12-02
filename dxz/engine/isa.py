from torch import Tensor
from typing import Optional

class Instruction:
    pass

class Fill(Instruction):
    def __init__(self, token_ids: Optional[list[int]], position_ids: list[int], cache_ids: list[list[int]], kv_cache_ids: list[int], sample: bool, sample_dst: Optional["Fill"]):
        super().__init__()
        # cache_ids (n_layers, n_tokens)
        # kv_caches (n_layers, )
        self.token_ids = token_ids
        self.position_ids = position_ids
        self.cache_ids = cache_ids
        self.kv_cache_ids = kv_cache_ids
        self.sample = sample
        self.sample_dst = sample_dst

class TextFill(Fill):
    def __init__(self, token_ids: Optional[list[int]], position_ids: list[int], cache_ids: list[list[int]], kv_cache_ids: list[int], sample: bool, sample_dst: Optional[Fill]):
        super().__init__(token_ids, position_ids, cache_ids, kv_cache_ids, sample, sample_dst)

    def __repr__(self):
        return f"TextFill {self.token_ids} {self.position_ids} {self.cache_ids} {self.kv_cache_ids}"

class ImageFill(Fill):
    def __init__(self, pixel_values: Tensor, token_ids: Optional[list[int]], position_ids: list[int], cache_ids: list[list[int]], kv_cache_ids: list[int], sample: bool, sample_dst: Optional[Fill]):
        super().__init__(token_ids, position_ids, cache_ids, kv_cache_ids, sample, sample_dst)
        self.pixel_values    = pixel_values # (n_images, n_channel, )

    def __repr__(self):
        return f"ImageFill {self.token_ids[:3]}...{self.token_ids[-3:]} {self.position_ids[:3]}...{self.position_ids[-3:]} {self.cache_ids[:3]}...{self.cache_ids[-3:]} {self.kv_cache_ids}"

class ImageEmbedFill(Fill):
    def __init__(self, image_featues: Optional[Tensor], token_ids: Optional[list[int]], position_ids: list[int], cache_ids: list[list[int]], kv_cache_ids: list[int], sample: bool, sample_dst: Optional[Fill]):
        super().__init__(token_ids, position_ids, cache_ids, kv_cache_ids, sample, sample_dst)
        self.image_features = image_featues # (m_tokens, hidden_size)

class Mov(Instruction):
    def __init__(self, src_cache_ids: list[list[int]], dst_cache_ids: list[list[int]], src_kv_cache_ids: list[int], dst_kv_cache_ids: list[int]):
        super().__init__()
        self.src_cache_ids = src_cache_ids
        self.dst_cache_ids = dst_cache_ids
        self.src_kv_cache_ids = src_kv_cache_ids
        self.dst_kv_cache_ids = dst_kv_cache_ids

    def __repr__(self):
        return f"Mov {self.src_cache_ids} {self.dst_cache_ids} {self.src_cache_ids} {self.dst_cache_ids}"

class ReAlloc(Instruction):
    def __init__(self, n_tokens: list[int], kv_cache_ids: list[int]):
        self.n_tokens     = n_tokens
        self.kv_cache_ids = kv_cache_ids

    def __repr__(self):
        return f"ReAlloc {self.n_tokens} {self.kv_cache_ids}"

class Merge(Instruction):
    def __init__(self, kv_cache1_ids: list[int], kv_cache2_ids: list[int]):
        super().__init__()
        self.kv_cache1_ids = kv_cache1_ids
        self.kv_cache2_ids = kv_cache2_ids

class EmptyInstruction(Instruction):
    pass

class ImageEmbed(Instruction):
    def __init__(self, pixel_values: Tensor, image_featues_dst: Optional[ImageEmbedFill], token_pruning_params: dict):
        super().__init__()
        self.pixel_values = pixel_values
        self.image_featues_dst = image_featues_dst
        self.token_pruning_params = token_pruning_params