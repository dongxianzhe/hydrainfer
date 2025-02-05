from torch import Tensor
from typing import Optional


class Instruction:
    next: "Instruction" = None
    prev: "Instruction" = None

    def insert_next(self, inst: "Instruction"):
        inst.next = self.next
        inst.prev = self
        self.next.prev = inst
        self.next = inst


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

    def chunk_prefill(self, chunk_size: int):
        assert chunk_size > 0 and chunk_size < len(self.token_ids), f"invalid chunk prefill size {chunk_size}"
        rest_text_fill = TextFill(
            token_ids = self.token_ids[chunk_size:], 
            position_ids = self.position_ids[chunk_size:], 
            cache_ids = [layer_cahce_ids[chunk_size:] for layer_cahce_ids in self.cache_ids], 
            kv_cache_ids = self.kv_cache_ids, 
            sample = self.sample, 
            sample_dst = self.sample_dst, 
        )
        self.insert_next(rest_text_fill)
        self.token_ids = self.token_ids[:chunk_size]
        self.position_ids = self.position_ids[:chunk_size]
        self.cache_ids = [layer_cahce_ids[:chunk_size] for layer_cahce_ids in self.cache_ids]
        self.sample = False
        self.sample_dst = None

    def __repr__(self):
        return "TF"


class ImageFill(Fill):
    def __init__(self, pixel_values: Tensor, token_ids: Optional[list[int]], position_ids: list[int], cache_ids: list[list[int]], kv_cache_ids: list[int], sample: bool, sample_dst: Optional[Fill]):
        super().__init__(token_ids, position_ids, cache_ids, kv_cache_ids, sample, sample_dst)
        self.pixel_values    = pixel_values # (n_images, n_channel, )

    def __repr__(self):
        return "IF"


class ImageEmbedFill(Fill):
    def __init__(self, image_features: Optional[Tensor], token_ids: Optional[list[int]], position_ids: list[int], cache_ids: list[list[int]], kv_cache_ids: list[int], sample: bool, sample_dst: Optional[Fill]):
        super().__init__(token_ids, position_ids, cache_ids, kv_cache_ids, sample, sample_dst)
        self.image_features = image_features # (m_tokens, hidden_size)

    def __repr__(self):
        return "EF"


    def chunk_prefill(self, chunk_size: int):
        rest_text_fill = ImageEmbedFill(
            image_features = self.image_features[chunk_size:, ...], 
            token_ids = self.token_ids[chunk_size:], 
            position_ids = self.position_ids[chunk_size:], 
            cache_ids = [layer_cahce_ids[chunk_size:] for layer_cahce_ids in self.cache_ids], 
            kv_cache_ids = self.kv_cache_ids, 
            sample = self.sample, 
            sample_dst = self.sample_dst, 
        )
        self.image_features = self.image_features[:chunk_size, ...],  
        self.insert_next(rest_text_fill)
        self.token_ids = self.token_ids[:chunk_size]
        self.position_ids = self.position_ids[:chunk_size]
        self.cache_ids = [layer_cahce_ids[:chunk_size] for layer_cahce_ids in self.cache_ids]
        self.sample = False
        self.sample_dst = None


class Mov(Instruction):
    def __init__(self, src_cache_ids: list[list[int]], dst_cache_ids: list[list[int]], src_kv_cache_ids: list[int], dst_kv_cache_ids: list[int]):
        super().__init__()
        self.src_cache_ids = src_cache_ids
        self.dst_cache_ids = dst_cache_ids
        self.src_kv_cache_ids = src_kv_cache_ids
        self.dst_kv_cache_ids = dst_kv_cache_ids

    def __repr__(self):
        return "MV"


class ReAlloc(Instruction):
    def __init__(self, n_tokens: list[int], kv_cache_ids: list[int]):
        self.n_tokens     = n_tokens
        self.kv_cache_ids = kv_cache_ids

    def __repr__(self):
        return "RA"


class Merge(Instruction):
    def __init__(self, kv_cache1_ids: list[int], kv_cache2_ids: list[int]):
        super().__init__()
        self.kv_cache1_ids = kv_cache1_ids
        self.kv_cache2_ids = kv_cache2_ids

    def __repr__(self):
        return "MR"


class EmptyInstruction(Instruction):
    def __repr__(self):
        return "EM"


class ImageEmbed(Instruction):
    def __init__(self, pixel_values: Tensor, image_features_dst: Optional[ImageEmbedFill], token_pruning_params: dict):
        super().__init__()
        self.pixel_values = pixel_values
        self.image_features_dst = image_features_dst
        self.token_pruning_params = token_pruning_params

    def __repr__(self):
        return "IE"


class MigrateRequest(Instruction):
    def __repr__(self):
        return "MR"


class InstructionList:
    def __init__(self, head: Instruction, tail: Instruction, curr: Instruction):
        self.head = head
        self.tail = tail
        self.curr = curr

    def __repr__(self):
        s = ""
        curr = self.head
        while curr:
            if curr is self.curr:
                s += "*"
            s += repr(curr)
            if curr.next is not None:
                s += "->"
            curr = curr.next
        return s


class InstructionListBuilder:
    def __init__(self):
        self.head = EmptyInstruction()
        self.tail = EmptyInstruction()
        self.head.next = self.tail
        self.head.prev = None
        self.tail.prev = self.head
        self.tail.next = None

    def append(self, inst: Instruction):
        inst.prev = self.tail.prev
        inst.next = self.tail
        self.tail.prev.next = inst
        self.tail.prev = inst

    def build_instruction_list(self) -> InstructionList:
        return InstructionList(self.head, self.tail, self.head.next)