import random
from torch import Tensor
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoProcessor
from dxz.engine.isa import Instruction, Fill, ImageFill, Mov, ReAlloc
from PIL import Image
from typing import Literal

@dataclass
class CompilerConfig:
    tokenizer: AutoTokenizer
    processor: AutoProcessor
    image_token_id: int
    num_image_tokens: int # number of tokens each image embedding
    n_layers: int
    kv_cache_eviction_policy: Literal['vanilla', 'random', 'streamingllm'] = "random"
    # streamingLLM params
    window_size: int = 12
    attention_sink_size: int = 4

class Compiler:
    def __init__(self, config: CompilerConfig):
        self.config = config
        self.tokenizer = self.config.tokenizer
        self.processor = self.config.processor
        self.image_token_id = self.config.image_token_id
        self.num_image_tokens = self.config.num_image_tokens
        if config.kv_cache_eviction_policy == 'vanilla':
            self.code_generator = VanillaCodeGenerator(config)
        elif config.kv_cache_eviction_policy == 'random':
            self.code_generator = RandomCodeGenerator(config)
        elif config.kv_cache_eviction_policy == 'streamingllm':
            self.code_generator = StreamingLLMCodeGenerator(config)

    def tokenize(self, prompt: str, images: list[Image.Image]) -> tuple[list[int], list[Tensor]]:
        token_ids = self.tokenizer.encode(prompt)

        inserted_token_ids = []
        for token_id in token_ids:
            if token_id == self.image_token_id:
                inserted_token_ids.extend([self.image_token_id] * (self.num_image_tokens - 1))
            inserted_token_ids.append(token_id)
        
        images = [self.processor(
            text="", 
            images=image, 
            return_tensors="pt"
        )['pixel_values'] for image in images]
        return inserted_token_ids, images

    def code_generate(self, token_ids: list[int], pixel_values: list[Tensor]) -> tuple[list[Instruction], int]:
        return self.code_generator.code_generate(token_ids, pixel_values)

    def compile(self, prompt: str, images: list[Image.Image]) -> list[Instruction]:
        token_ids, pixel_values = self.tokenize(prompt, images)
        instructions, n_virtual_kv_caches =  self.code_generate(token_ids, pixel_values)
        return instructions, n_virtual_kv_caches

    def interpret_next_instruction(self, params: "DecodeParams") -> Instruction:
        return self.code_generator.interpret_next_instruction(params)

class DecodeParams:
    def __init__(self, curr_instruction: Instruction, next_token_id: int):
        self.curr_instruction = curr_instruction
        self.next_token_id = next_token_id

class VanillaCodeGenerator:
    """ 
        full attention 
        no chunk prefill
    """
    def __init__(self, config: CompilerConfig):
        self.config = config

    def code_generate(self, token_ids: list[int], pixel_values: list[Tensor]) -> tuple[list[Instruction], int]:
        instructions: list[Instruction] = []
        instructions.append(ImageFill(
            images = pixel_values, 
            token_ids = token_ids,
            position_ids = list(range(0, len(token_ids))), 
            cache_ids = [list(range(0, len(token_ids))) for _ in range(self.config.n_layers)], 
            kv_cache_ids = list(range(self.config.n_layers)), 
            sample = True, 
        ))
        n_virtual_kv_caches = self.config.n_layers
        return instructions, n_virtual_kv_caches

    def interpret_next_instruction(self, params: DecodeParams) -> Instruction:
        return Fill(
            token_ids = [params.next_token_id],
            position_ids = [params.curr_instruction.position_ids[-1] + 1], 
            cache_ids = [[layer_cache_ids[-1] + 1] for layer_cache_ids in params.curr_instruction.cache_ids], 
            kv_cache_ids = params.curr_instruction.kv_cache_ids,
            sample = True
            )

class MultiModalChunkPrefillCodeGenerator:
    """ 
        full attention 
        image and text seperate chunk prefill
    """
    def __init__(self, config: CompilerConfig):
        self.config = config

    def code_generate(self, token_ids: list[int], pixel_values: list[Tensor]) -> tuple[list[Instruction], int]:
        instructions: list[Instruction] = []
        i = 0
        while i < len(token_ids):
            if token_ids[i] == self.config.image_token_id:
                j = i
                while j < len(token_ids) and token_ids[j] == self.config.image_token_id:
                    j += 1
                instructions.append(ImageFill(
                    images = pixel_values, 
                    token_ids = token_ids[i:j],
                    position_ids = list(range(i, j)), 
                    cache_ids = [list(range(i, j)) for _ in range(self.config.n_layers)], 
                    kv_cache_ids = list(range(self.config.n_layers)), 
                    sample = j==len(token_ids), 
                ))
                i = j
            else:
                j = i
                while j < len(token_ids) and token_ids[j] != self.config.image_token_id:
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

    def interpret_next_instruction(self, params: DecodeParams) -> Instruction:
        return Fill(
            token_ids = [params.next_token_id],
            position_ids = [params.curr_instruction.position_ids[-1] + 1], 
            cache_ids = [[layer_cache_ids[-1] + 1] for layer_cache_ids in params.curr_instruction.cache_ids], 
            kv_cache_ids = params.curr_instruction.kv_cache_ids,
            sample = True
            )

class StreamingLLMCodeGenerator:
    def __init__(self, config: CompilerConfig):
        self.config = config
        self.window_size = self.config.window_size
        self.attention_sink_size = self.config.attention_sink_size

    def code_generate(self, token_ids: list[int], pixel_values: list[Tensor]) -> tuple[list[Instruction], int]:
        instructions: list[Instruction] = []
        n_virtual_kv_caches = self.config.n_layers
        virtual_kv_cache_ids = list(range(n_virtual_kv_caches))

        n_prompt_tokens = len(token_ids)
        # 1. prefill
        instructions.append(ImageFill(
            images = pixel_values, 
            token_ids = token_ids,
            position_ids = list(range(0, n_prompt_tokens)), 
            cache_ids = [list(range(0, n_prompt_tokens)) for _ in range(n_virtual_kv_caches)], 
            kv_cache_ids = virtual_kv_cache_ids, 
            sample = True, 
        ))
        # 2. kv cache eviction
        cache_size = self.attention_sink_size + self.window_size
        print(f'n_prompt_tokens {n_prompt_tokens}')
        print(f'cache_size {cache_size}')
        if n_prompt_tokens > cache_size:
            instructions.append(Mov(
                src_cache_ids = [list(range(n_prompt_tokens - self.window_size, n_prompt_tokens)) for _ in range(n_virtual_kv_caches)], 
                dst_cache_ids = [list(range(self.attention_sink_size, self.attention_sink_size + self.window_size)) for _ in range(n_virtual_kv_caches)], 
                src_kv_cache_ids = virtual_kv_cache_ids, 
                dst_kv_cache_ids = virtual_kv_cache_ids, 
            ))
            instructions.append(ReAlloc(
                n_tokens = [cache_size for _ in range(n_virtual_kv_caches)], 
                kv_cache_ids = virtual_kv_cache_ids, 
            ))

        for instruction in instructions:
            print(instruction)

        return instructions, n_virtual_kv_caches

    def interpret_next_instruction(self, params: DecodeParams) -> Instruction:
        cache_ids: list[int] = []
        for layer_cache_ids in params.curr_instruction.cache_ids:
            next_token_id = params.next_token_id
            cache_id = layer_cache_ids[-1]
            window_offset = cache_id - self.attention_sink_size
            next_cache_id = (window_offset + 1) % self.window_size + self.attention_sink_size
            cache_ids.append([next_cache_id])

        instruction = Fill(
            token_ids = [params.next_token_id],
            position_ids = [params.curr_instruction.position_ids[-1] + 1], 
            cache_ids = cache_ids, 
            kv_cache_ids = params.curr_instruction.kv_cache_ids,
            sample = True
            )

        print(instruction)

        return instruction

class RandomCodeGenerator:
    def __init__(self, config: CompilerConfig):
        self.config = config
        self.window_size = self.config.window_size
        self.attention_sink_size = self.config.attention_sink_size

    def code_generate(self, token_ids: list[int], pixel_values: list[Tensor]) -> tuple[list[Instruction], int]:
        instructions: list[Instruction] = []
        n_virtual_kv_caches = self.config.n_layers
        virtual_kv_cache_ids = list(range(n_virtual_kv_caches))
        n_prompt_tokens = len(token_ids)
        instructions.append(ImageFill(
            images = pixel_values, 
            token_ids = token_ids,
            position_ids = list(range(0, n_prompt_tokens)), 
            cache_ids = [list(range(0, n_prompt_tokens)) for _ in range(n_virtual_kv_caches)], 
            kv_cache_ids = virtual_kv_cache_ids, 
            sample = True, 
        ))

        return instructions, n_virtual_kv_caches

    def interpret_next_instruction(self, params: DecodeParams) -> Instruction:
        cache_ids: list[int] = []
        for _ in range(len(params.curr_instruction.cache_ids)):
            next_cache_id = random.randint(0, 576)
            cache_ids.append([next_cache_id])

        instruction = Fill(
            token_ids = [params.next_token_id],
            position_ids = [params.curr_instruction.position_ids[-1] + 1], 
            cache_ids = cache_ids, 
            kv_cache_ids = params.curr_instruction.kv_cache_ids,
            sample = True
            )

        return instruction