from dataclasses import dataclass
from transformers import AutoTokenizer, AutoProcessor
from dxz.engine.isa import Instruction, Fill, ImageFill
from PIL import Image

@dataclass
class CompilerConfig:
    tokenizer: AutoTokenizer
    processor: AutoProcessor
    image_token_id: int
    num_image_tokens: int # number of tokens each image embedding
    n_layers: int

class Compiler:
    def __init__(self, config: CompilerConfig):
        self.config = config
        self.tokenizer = self.config.tokenizer
        self.processor = self.config.processor
        self.image_token_id = self.config.image_token_id
        self.num_image_tokens = self.config.num_image_tokens

    def tokenize(self, prompt: str) -> list[int]:
        token_ids = self.tokenizer.encode(prompt)

        inserted_token_ids = []
        for token_id in token_ids:
            if token_id == self.image_token_id:
                inserted_token_ids.extend([self.image_token_id] * (self.num_image_tokens - 1))
            inserted_token_ids.append(token_id)
        
        return inserted_token_ids

    def code_generate(self, token_ids: list[int], images: list[Image.Image]) -> tuple[list[Instruction], int]:
        images = [self.processor(
            text="", 
            images=image, 
            return_tensors="pt"
        )['pixel_values'] for image in images]

        instructions: list[Instruction] = []
        i = 0
        while i < len(token_ids):
            if token_ids[i] == self.image_token_id:
                j = i
                while j < len(token_ids) and token_ids[j] == self.image_token_id:
                    j += 1
                instructions.append(ImageFill(
                    images = images, 
                    token_ids = token_ids[i:j],
                    position_ids = list(range(i, j)), 
                    cache_ids = [list(range(i, j)) for _ in range(self.config.n_layers)], 
                    kv_cache_ids = list(range(self.config.n_layers)), 
                    sample = j==len(token_ids), 
                ))
                i = j
            else:
                j = i
                while j < len(token_ids) and token_ids[j] != self.image_token_id:
                    j += 1
                instructions.append(Fill(
                    token_ids = token_ids[i:j], 
                    position_ids = list(range(i, j)), 
                    cache_ids = [list(range(i, j)) for _ in range(self.config.n_layers)], 
                    kv_cache_ids = list(range(self.config.n_layers)), 
                    sample = j==len(token_ids), 
                ))
                i = j 

        n_virtual_kv_caches = self.config.n_layers
        return instructions, n_virtual_kv_caches

    def compile(self, prompt: str, images: list[Image.Image]) -> list[Instruction]:
        token_ids = self.tokenize(prompt)
        instructions, n_virtual_kv_caches =  self.code_generate(token_ids, images)
        return instructions, n_virtual_kv_caches

    def interpret_next_instruction(self, instruction: Instruction, token_id: int) -> Instruction:
        next_instruction = Fill(
            token_ids = [token_id],
            position_ids = [instruction.position_ids[-1] + 1], 
            cache_ids = [[layer_cache_ids[-1] + 1] for layer_cache_ids in instruction.cache_ids], 
            kv_cache_ids = instruction.kv_cache_ids,
            sample = True
            )
        return next_instruction