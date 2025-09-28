from torch import Tensor
from typing import Optional, Literal


class Instruction:
    """
        Instruction is the scheduling granularity of the batch scheduler
        for a typical request the instruction linked list will be 
        ImageEmbed -> EPMigrate -> PullCache -> Fill -> PDMigrate -> PullCache -> Fill -> ...(many decode Fill) -> Empty
    """
    next: "Instruction" = None
    prev: "Instruction" = None

    def insert_next(self, inst: "Instruction"):
        inst.next = self.next
        inst.prev = self
        self.next.prev = inst
        self.next = inst


class Fill(Instruction):
    """
    we use fill instruction to implement prefill and decode stage
    for example if we have a prompt "who are you ?" and it is tokenized as [1, 2, 3, 4]
    then prefill stage
    token_ids  [1, 2, 3, 4]
    position_ids is [0, 1, 2, 3]
    cache_ids is [0, 1, 2, 3]
    sample is True
    sample dst is next Fill instruction used in decode stage

    decode Fill's token_ids is None when request created and set when autoregressing
    """
    def __init__(
        self, 
        token_ids: Optional[list[int]],
        position_ids: list[int], 
        cache_ids: list[int], 
        sample: bool, 
        sample_dst: Optional["Fill"]
    ):
        super().__init__()
        self.token_ids = token_ids
        self.position_ids = position_ids
        self.cache_ids = cache_ids
        self.sample = sample
        self.sample_dst = sample_dst
        self.is_chunked = False # used to control weather return sample token to api server


class TextFill(Fill):
    def __init__(
        self, 
        token_ids: Optional[list[int]], 
        position_ids: list[int], 
        cache_ids: list[int], 
        sample: bool, 
        sample_dst: Optional[Fill]
    ):
        super().__init__(token_ids, position_ids, cache_ids, sample, sample_dst)

    def chunk_prefill(self, chunk_size: int):
        """
        chunked prefill must at least contain one token and less than all tokens
        """
        assert chunk_size > 0 and chunk_size < len(self.token_ids), f"invalid chunk prefill size {chunk_size}"
        rest_text_fill = TextFill(
            token_ids = self.token_ids[chunk_size:], 
            position_ids = self.position_ids[chunk_size:], 
            cache_ids = self.cache_ids[chunk_size:], 
            sample = self.sample, 
            sample_dst = self.sample_dst, 
        )
        self.insert_next(rest_text_fill)
        self.token_ids = self.token_ids[:chunk_size]
        self.position_ids = self.position_ids[:chunk_size]
        self.cache_ids = self.cache_ids[:chunk_size]
        self.sample = True # todo remove the unnecessary sample. when a batch has no sample request because of chunked prefill, do not know why model forward will crash?.
        self.sample_dst = EmptyInstruction()
        self.is_chunked = True

    def __repr__(self):
        return "TF"


class ImageEmbedFill(Fill):
    """
    for example if we have a prompt "<image> what's the content of this image"
    and tokenized as [x, x, x, x, what's, the, content, of, this, image]
    then image_token_cache_ids is [0, 1, 2, 3]
    image_token_mask [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    """
    def __init__(self, 
        image_token_cache_ids: list[int], 
        image_token_mask: list[bool], 
        token_ids: Optional[list[int]], 
        position_ids: list[int], 
        cache_ids: list[int], 
        sample: bool, 
        sample_dst: Optional[Fill]
    ):
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
            token_ids = self.token_ids[chunk_size:], 
            position_ids = self.position_ids[chunk_size:], 
            cache_ids = self.cache_ids[chunk_size:], 
            sample = self.sample, 
            sample_dst = self.sample_dst, 
        )
        self.insert_next(rest_fill)
        self.image_token_cache_ids = self.image_token_cache_ids[:num_image_token]
        self.image_token_mask = self.image_token_mask[:chunk_size]
        self.token_ids = self.token_ids[:chunk_size]
        self.position_ids = self.position_ids[:chunk_size]
        self.cache_ids = self.cache_ids[:chunk_size]
        self.sample = True # todo remove the unnecessary sample. when a batch has no sample request because of chunked prefill, do not know why model forward will crash?.
        self.sample_dst = EmptyInstruction()
        self.is_chunked = True


class EmptyInstruction(Instruction):
    def __repr__(self):
        return "EM"


class ImageEmbed(Instruction):
    def __init__(
        self, 
        pixel_values: Tensor, 
        cache_ids: list[int], 
        images_size: list[tuple[int, int]], 
    ):
        super().__init__()
        self.pixel_values = pixel_values
        self.cache_ids = cache_ids
        self.images_size = images_size

    def __repr__(self):
        return "IE"


class MigrateRequest(Instruction):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return f"MR"


class EPMigrate(MigrateRequest):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return f"EPMR"
    
class PDMigrate(MigrateRequest):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return f"PDMR"


class PullCache(Instruction):
    def __repr__(self):
        return f"PR"


class InstructionList:
    """ we use double linked list with empty head and empty tail"""
    def __init__(
        self, 
        head: Instruction, 
        tail: Instruction, 
        curr: Instruction
    ):
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