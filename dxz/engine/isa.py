from torch import Tensor
from typing import Optional


class Instruction:
    """
        Instruction is the scheduling granularity of the batch scheduler
    """
    next: "Instruction" = None
    prev: "Instruction" = None

    def insert_next(self, inst: "Instruction"):
        inst.next = self.next
        inst.prev = self
        self.next.prev = inst
        self.next = inst


class Fill(Instruction):
    def __init__(self, token_ids: Optional[list[int]], position_ids: list[int], cache_ids: list[int], sample: bool, sample_dst: Optional["Fill"]):
        super().__init__()
        # cache_ids (n_layers, n_tokens)
        # kv_caches (n_layers, )
        self.token_ids = token_ids
        self.position_ids = position_ids
        self.cache_ids = cache_ids
        self.sample = sample
        self.sample_dst = sample_dst


class TextFill(Fill):
    def __init__(self, token_ids: Optional[list[int]], position_ids: list[int], cache_ids: list[int], sample: bool, sample_dst: Optional[Fill]):
        super().__init__(token_ids, position_ids, cache_ids, sample, sample_dst)

    def chunk_prefill(self, chunk_size: int):
        assert chunk_size > 0 and chunk_size < len(self.token_ids), f"invalid chunk prefill size {chunk_size}"
        rest_text_fill = TextFill(
            token_ids = self.token_ids[chunk_size:], 
            position_ids = self.position_ids[chunk_size:], 
            cache_ids = self.cache_ids[chunk_size:], 
            kv_cache_ids = self.kv_cache_ids, 
            sample = self.sample, 
            sample_dst = self.sample_dst, 
        )
        self.insert_next(rest_text_fill)
        self.token_ids = self.token_ids[:chunk_size]
        self.position_ids = self.position_ids[:chunk_size]
        self.cache_ids = self.cache_ids[:chunk_size]
        self.sample = False
        self.sample_dst = None

    def __repr__(self):
        return "TF"


class ImageFill(Fill):
    def __init__(self, pixel_values: Tensor, token_ids: Optional[list[int]], position_ids: list[int], cache_ids: list[int], sample: bool, sample_dst: Optional[Fill]):
        super().__init__(token_ids, position_ids, cache_ids, sample, sample_dst)
        self.pixel_values    = pixel_values # (n_images, n_channel, )

    def __repr__(self):
        return "IF"


class ImageEmbedFill(Fill):
    def __init__(self, image_token_cache_ids: list[int], image_token_mask: list[bool], token_ids: Optional[list[int]], position_ids: list[int], cache_ids: list[int], sample: bool, sample_dst: Optional[Fill]):
        super().__init__(token_ids, position_ids, cache_ids, sample, sample_dst)
        self.image_token_cache_ids = image_token_cache_ids # 0, 1, 2, ..., 575 
        self.image_token_mask = image_token_mask

    def __repr__(self):
        return "EF"

    def chunk_prefill(self, chunk_size: int):
        assert chunk_size > 0 and chunk_size < len(self.token_ids), f"invalid chunk prefill size {chunk_size}"
        image_token_mask_chunk = self.image_token_mask[:chunk_size]
        num_image_token = sum(image_token_mask_chunk)
        rest_fill = ImageEmbedFill(
            image_token_cache_ids = self.image_token_cache_ids[num_image_token:], 
            image_token_mask = self.image_token_mask[chunk_size:], 
            image_features = self.image_features[chunk_size:, ...], 
            token_ids = self.token_ids[chunk_size:], 
            position_ids = self.position_ids[chunk_size:], 
            cache_ids = self.cache_ids[chunk_size:], 
            kv_cache_ids = self.kv_cache_ids, 
            sample = self.sample, 
            sample_dst = self.sample_dst, 
        )
        self.insert_next(rest_fill)
        self.image_token_cache_ids = self.image_token_cache_ids[:num_image_token]
        self.image_token_mask = self.image_token_mask[:chunk_size]
        self.token_ids = self.token_ids[:chunk_size]
        self.position_ids = self.position_ids[:chunk_size]
        self.cache_ids = self.cache_ids[:chunk_size]
        self.sample = False
        self.sample_dst = None


class EmptyInstruction(Instruction):
    def __repr__(self):
        return "EM"


class ImageEmbed(Instruction):
    def __init__(self, pixel_values: Tensor, cache_ids: list[int], token_pruning_params: dict):
        super().__init__()
        self.pixel_values = pixel_values
        self.cache_ids = cache_ids
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