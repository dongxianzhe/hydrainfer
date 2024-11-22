from torch import Tensor

class Instruction:
    def __init__(self):
        pass

class Prefill(Instruction):
    def __init__(self, token_ids: list[int]):
        super().__init__()
        self.token_ids = token_ids

class BlockPrefill(Instruction):
    def __init__(self, token_ids: list[int]):
        super().__init__()
        self.token_ids = token_ids 

class ImagePrefill(Instruction):
    def __init__(self, token_ids: list[int], pixel_values: Tensor):
        super().__init__()
        self.token_ids = token_ids
        self.pixel_values = pixel_values

class ImageBlockPrefill(Instruction):
    def __init__(self, token_ids: list[int], pixel_values: Tensor):
        super().__init__()
        self.token_ids = token_ids
        self.pixel_values = pixel_values

class Decode(Instruction):
    def __init__(self, token_id: int):
        super().__init__()
        self.token_id = token_id 

class Return(Instruction):
    def __init__(self):
        pass

class Compiler:
    def __init__(self, image_token_id: int):
        self.image_token_id = image_token_id

    def compile(self, token_ids: list[int], pixel_values: Tensor) -> list[Instruction]:
        instructions: list[Instruction] = []
        i = 0
        while i < len(token_ids):
            if token_ids[i] == self.image_token_id:
                j = i
                while j < len(token_ids) and token_ids[j] == self.image_token_id:
                    j += 1
                instructions.append(ImagePrefill(token_ids[i:j], pixel_values))
                i = j
            else:
                j = i
                while j < len(token_ids) and token_ids[j] != self.image_token_id:
                    j += 1
                instructions.append(Prefill(token_ids[i:j]))
                i = j
        
        instructions.append(Return())
        return instructions