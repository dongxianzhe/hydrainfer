import random
from torch import Tensor
from dataclasses import dataclass, fields
from transformers import AutoTokenizer, AutoProcessor
from dxz.engine.isa import Instruction, TextFill, ImageFill, Mov, ReAlloc, EmptyInstruction, ImageEmbed, ImageEmbedFill
from PIL import Image
from typing import Literal, Optional
import argparse

@dataclass
class CompilerConfig:
    default_max_tokens: int = 64
    disaggregate_embed_prefill: bool = True
    # chunked_prefill: bool = True
    # max_chunk_size: int = 256
    kv_cache_eviction_policy: Literal[None, 'random', 'streamingllm'] = None
    window_size: int = 28
    attention_sink_size: int = 4 
    token_pruning_policy: Literal[None, 'focal'] = None
    n_embed_output_tokens: int = 64

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> 'CompilerConfig':
        attrs = [attr.name for attr in fields(cls)]
        config = cls(**{attr: getattr(args, attr) for attr in attrs})
        return config

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--default-max-tokens', type=int, default=64, help='Maximum default number of output tokens.')
        parser.add_argument('--disaggregate-embed-prefill', action='store_true', help='Enable disaggregation of embedding prefill.')
        parser.add_argument('--kv-cache-eviction-policy', type=str, choices=[None, 'random', 'streamingllm'], default=None, help='Eviction policy for key-value cache.')
        parser.add_argument('--window-size', type=int, default=28, help='Size of the attention window.')
        parser.add_argument('--attention-sink-size', type=int, default=4, help='Size of the attention sink.')
        parser.add_argument('--token-pruning-policy', type=str, choices=[None, 'focal'], default=None, help='Token pruning policy.')
        parser.add_argument('--n-embed-output-tokens', type=int, default=64, help='Number of output tokens for embedding.')
        return parser

@dataclass
class CompilerContext:
    tokenizer: AutoTokenizer
    processor: AutoProcessor
    image_token_id: int
    num_image_tokens: int
    n_layers: int

@dataclass
class CompileParameters:
    max_tokens: Optional[int] = None

@dataclass
class CompilerOutput:
    instructions: list[Instruction]
    n_virtual_kv_caches: int
    n_prompt_tokens: int


class Compiler:
    def __init__(self, config: CompilerConfig, context: CompilerContext):
        self.config = config
        self.context = context
        self.tokenizer = context.tokenizer
        self.processor = context.processor
        self.image_token_id = self.context.image_token_id
        self.num_image_tokens = self.context.num_image_tokens

    def tokenize(self, prompt: str, images: list[Image.Image], compile_params: CompileParameters) -> tuple[list[int], list[Tensor]]:
        token_ids = self.tokenizer.encode(prompt)
        num_image_tokens = self.config.n_embed_output_tokens if self.config.token_pruning_policy else self.num_image_tokens
        inserted_token_ids = []
        for token_id in token_ids:
            if token_id == self.image_token_id:
                inserted_token_ids.extend([self.image_token_id] * (num_image_tokens - 1))
            inserted_token_ids.append(token_id)
        images = self.processor(
            text="", 
            images=images, 
            return_tensors="pt"
        )['pixel_values'] # (n_images, n_channels, width, height)
        return inserted_token_ids, images

    def code_generate(self, token_ids: list[int], pixel_values: Tensor, compile_param: CompileParameters) -> tuple[list[Instruction], int]:
        instructions: list[Instruction] = []
        n_virtual_kv_caches: int = self.context.n_layers

        layer_virtual_kv_cache_ids = list(range(self.context.n_layers))
        if self.config.disaggregate_embed_prefill:
            # 1. embed (token pruning)
            embed_instruction = ImageEmbed(
                pixel_values = pixel_values, 
                image_featues_dst = None, 
                token_pruning_params = {'policy': self.config.token_pruning_policy, 'n_output_tokens': self.config.n_embed_output_tokens}, 
            )
            instructions.append(embed_instruction)
            # 2. prefill
            last_prefill_instruction = ImageEmbedFill(
                image_featues = None,
                token_ids = token_ids,
                position_ids = list(range(0, len(token_ids))), 
                cache_ids = [list(range(0, len(token_ids))) for _ in range(self.context.n_layers)], 
                kv_cache_ids = layer_virtual_kv_cache_ids, 
                sample = True, 
                sample_dst = None, 
            )
            instructions.append(last_prefill_instruction)
            embed_instruction.image_featues_dst = last_prefill_instruction
        else:
            assert self.config.token_pruning_policy == None, 'token pruning is not supported without disaggregate_embed_prefill'
            fill_inst = ImageFill(
                pixel_values = pixel_values, 
                token_ids = token_ids,
                position_ids = list(range(0, len(token_ids))), 
                cache_ids = [list(range(0, len(token_ids))) for _ in range(self.context.n_layers)],
                kv_cache_ids = layer_virtual_kv_cache_ids, 
                sample = True, 
                sample_dst= None,
            )
            instructions.append(fill_inst)
            last_prefill_instruction = fill_inst

        # 3. decode (kv_cache eviction)
        max_tokens = compile_param.max_tokens if compile_param.max_tokens is not None else self.config.default_max_tokens
        curr_fill_inst = last_prefill_instruction
        for _ in range(max_tokens - 1):
            cache_ids: list[int] = []
            if self.config.kv_cache_eviction_policy == 'random':
                for _ in range(self.context.n_layers):
                    next_cache_id = random.randint(0, len(token_ids)) # todo cache size
                    cache_ids.append([next_cache_id])
            elif self.config.kv_cache_eviction_policy == 'streamingllm':
                raise Exception('todo')
            else:
                for layer_id in range(self.context.n_layers):
                    cache_ids.append([curr_fill_inst.cache_ids[layer_id][-1] + 1])

            instruction = TextFill(
                token_ids = None,
                position_ids = [curr_fill_inst.position_ids[-1] + 1], 
                cache_ids = cache_ids, 
                kv_cache_ids = layer_virtual_kv_cache_ids,
                sample = True, 
                sample_dst= None,
            )
            curr_fill_inst.sample_dst = instruction
            instructions.append(instruction)
            curr_fill_inst = instruction
        tail_instruction = EmptyInstruction()
        curr_fill_inst.sample_dst = tail_instruction
        instructions.append(tail_instruction)

        return instructions, n_virtual_kv_caches

    def compile(self, prompt: str, images: list[Image.Image], compile_params: CompileParameters) -> CompilerOutput:
        token_ids, pixel_values = self.tokenize(prompt, images, compile_params)
        instructions, n_virtual_kv_caches = self.code_generate(token_ids, pixel_values, compile_params)
        return CompilerOutput(
            instructions = instructions, 
            n_virtual_kv_caches = n_virtual_kv_caches, 
            n_prompt_tokens = len(token_ids), 
        )

class VanillaCodeGenerator:
    """ 
        full attention 
        no chunk prefill
    """
    def __init__(self, config: CompilerConfig, context: CompilerContext):
        self.config = config
        self.context = context

    def code_generate(self, token_ids: list[int], pixel_values: Tensor) -> tuple[list[Instruction], int]:
        instructions: list[Instruction] = []
        instructions.append(ImageEmbed(
            pixel_values = pixel_values, 
            image_featues_dst = None, 
            token_pruning_params=None
        ))
        instructions.append(ImageEmbedFill(
            image_featues = None,
            token_ids = token_ids,
            position_ids = list(range(0, len(token_ids))), 
            cache_ids = [list(range(0, len(token_ids))) for _ in range(self.context.n_layers)], 
            kv_cache_ids = list(range(self.context.n_layers)), 
            sample = True, 
            sample_dst = None, 
        ))
        instructions[0].image_featues_dst = instructions[1]
        # instructions.append(ImageFill(
        #     pixel_values = pixel_values, 
        #     token_ids = token_ids,
        #     position_ids = list(range(0, len(token_ids))), 
        #     cache_ids = [list(range(0, len(token_ids))) for _ in range(self.context.n_layers)],
        #     kv_cache_ids = list(range(self.context.n_layers)), 
        #     sample = True, 
        #     sample_dst= None,
        # ))
        for _ in range(self.config.max_tokens):
            curr_instruction = instructions[-1]
            instructions.append(TextFill(
                token_ids = None,
                position_ids = [curr_instruction.position_ids[-1] + 1], 
                cache_ids = [[layer_cache_ids[-1] + 1] for layer_cache_ids in curr_instruction.cache_ids], 
                kv_cache_ids = curr_instruction.kv_cache_ids,
                sample = True, 
                sample_dst= None,
            ))
        instructions.append(EmptyInstruction())
        for i in range(len(instructions) - 1):
            instructions[i].sample_dst = instructions[i + 1]
        
        n_virtual_kv_caches = self.context.n_layers
        return instructions, n_virtual_kv_caches

class MultiModalChunkPrefillCodeGenerator:
    """ 
        full attention 
        image and text seperate chunk prefill
    """
    def __init__(self, config: CompilerConfig, context: CompilerContext):
        self.config = config
        self.context = context

    def code_generate(self, token_ids: list[int], pixel_values: Tensor) -> tuple[list[Instruction], int]:
        instructions: list[Instruction] = []
        i = 0
        while i < len(token_ids):
            if token_ids[i] == self.context.image_token_id:
                j = i
                while j < len(token_ids) and token_ids[j] == self.context.image_token_id:
                    j += 1
                instructions.append(ImageFill(
                    pixel_values = pixel_values, 
                    token_ids = token_ids[i:j],
                    position_ids = list(range(i, j)), 
                    cache_ids = [list(range(i, j)) for _ in range(self.context.n_layers)], 
                    kv_cache_ids = list(range(self.context.n_layers)), 
                    sample = j==len(token_ids), 
                    sample_dst= None,
                ))
                i = j
            else:
                j = i
                while j < len(token_ids) and token_ids[j] != self.context.image_token_id:
                    j += 1
                instructions.append(TextFill(
                    token_ids = token_ids[i:j], 
                    position_ids = list(range(i, j)), 
                    cache_ids = [list(range(i, j)) for _ in range(self.context.n_layers)], 
                    kv_cache_ids = list(range(self.context.n_layers)), 
                    sample = j==len(token_ids), 
                    sample_dst= None,
                ))
                i = j 

        for i in range(self.config.max_tokens):
            curr_instruction = instructions[-1]
            instructions.append(TextFill(
                token_ids = None, 
                position_ids = [curr_instruction.position_ids[-1] + 1], 
                cache_ids = [[layer_cache_ids[-1] + 1] for layer_cache_ids in curr_instruction.cache_ids], 
                kv_cache_ids = curr_instruction.kv_cache_ids,
                sample = True, 
                sample_dst= None,
            ))

        instructions.append(EmptyInstruction())

        for i in range(len(instructions) - 1):
            if instructions[i].sample:
                instructions[i].sample_dst = instructions[i + 1]

        n_virtual_kv_caches = self.context.n_layers
        return instructions, n_virtual_kv_caches

class StreamingLLMCodeGenerator:
    def __init__(self, config: CompilerConfig, context: CompilerContext):
        self.config = config
        self.context = context
        self.window_size = self.config.window_size
        self.attention_sink_size = self.config.attention_sink_size

    def code_generate(self, token_ids: list[int], pixel_values: Tensor) -> tuple[list[Instruction], int]:
        instructions: list[Instruction] = []
        n_virtual_kv_caches = self.context.n_layers
        virtual_kv_cache_ids = list(range(n_virtual_kv_caches))

        n_prompt_tokens = len(token_ids)
        # 1. prefill
        instructions.append(ImageFill(
            pixel_values = pixel_values, 
            token_ids = token_ids,
            position_ids = list(range(0, n_prompt_tokens)), 
            cache_ids = [list(range(0, n_prompt_tokens)) for _ in range(n_virtual_kv_caches)], 
            kv_cache_ids = virtual_kv_cache_ids, 
            sample = True, 
            sample_dst= None,
        ))
        curr_instruction = instructions[-1]
        # 2. kv cache eviction
        cache_size = self.attention_sink_size + self.window_size
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

        for i in range(self.config.max_tokens):
            cache_ids: list[int] = []
            for layer_cache_ids in curr_instruction.cache_ids:
                cache_id = layer_cache_ids[-1]
                window_offset = cache_id - self.attention_sink_size
                next_cache_id = (window_offset + 1) % self.window_size + self.attention_sink_size
                cache_ids.append([next_cache_id])

            instruction = TextFill(
                token_ids = None, 
                position_ids = [curr_instruction.position_ids[-1] + 1], 
                cache_ids = cache_ids, 
                kv_cache_ids = curr_instruction.kv_cache_ids,
                sample = True, 
                sample_dst= None,
                )
            curr_instruction.sample_dst = instruction
            instructions.append(instruction)
            curr_instruction = instruction
        instructions.append(EmptyInstruction())
        curr_instruction.sample_dst = instructions[-1]

        return instructions, n_virtual_kv_caches

class RandomCodeGenerator:
    def __init__(self, config: CompilerConfig, context: CompilerContext):
        self.config = config
        self.context = context
        self.window_size = self.config.window_size
        self.attention_sink_size = self.config.attention_sink_size

    def code_generate(self, token_ids: list[int], pixel_values: Tensor) -> tuple[list[Instruction], int]:
        instructions: list[Instruction] = []
        n_virtual_kv_caches = self.context.n_layers
        virtual_kv_cache_ids = list(range(n_virtual_kv_caches))
        n_prompt_tokens = len(token_ids)
        instructions.append(ImageFill(
            pixel_values = pixel_values, 
            token_ids = token_ids,
            position_ids = list(range(0, n_prompt_tokens)), 
            cache_ids = [list(range(0, n_prompt_tokens)) for _ in range(n_virtual_kv_caches)], 
            kv_cache_ids = virtual_kv_cache_ids, 
            sample = True, 
            sample_dst= None,
        ))


        for i in range(self.config.max_tokens):
            curr_instruction = instructions[-1]
            cache_ids: list[int] = []
            for _ in range(len(curr_instruction.cache_ids)):
                next_cache_id = random.randint(0, len(token_ids))
                cache_ids.append([next_cache_id])

            instructions.append(TextFill(
                token_ids = None, 
                position_ids = [curr_instruction.position_ids[-1] + 1], 
                cache_ids = cache_ids, 
                kv_cache_ids = curr_instruction.kv_cache_ids,
                sample = True, 
                sample_dst= None,
                ))
            
        instructions.append(EmptyInstruction())
        for i in range(len(instructions) - 1):
            if instructions[i].sample:
                instructions[i].sample_dst = instructions[i + 1]
            
        return instructions, n_virtual_kv_caches

class BlockPrefillLLMCodeGenerator():
    """ 
        block attention 
        image will only attend image when prefill
        text will only attend text when prefill
        decode state will attend all prompt
        image and text seperate chunk prefill
    """
    def __init__(self, config: CompilerConfig, context: CompilerContext):
        self.config = config
        self.context = context
        self.n_virtual_kv_caches = self.context.n_layers * 2
        self.image_kv_cache_ids = list(range(0, self.context.n_layers))
        self.text_kv_cache_ids  = list(range(self.context.n_layers, self.n_virtual_kv_caches))

    def code_generate(self, token_ids: list[int], pixel_values: Tensor) -> tuple[list[Instruction], int]:
        instructions: list[Instruction] = []
        n_text_tokens: int = 0
        n_image_tokens: int = 0
        i = 0
        while i < len(token_ids):
            if token_ids[i] == self.context.image_token_id:
                j = i
                while j < len(token_ids) and token_ids[j] == self.context.image_token_id:
                    j += 1
                instructions.append(ImageFill(
                    pixel_values = pixel_values, 
                    token_ids = token_ids[i:j],
                    position_ids = list(range(i, j)), 
                    cache_ids = [list(range(n_image_tokens, n_image_tokens + j - i)) for _ in range(self.context.n_layers)], 
                    kv_cache_ids = self.image_kv_cache_ids, 
                    sample = j==len(token_ids), 
                    sample_dst= None,
                ))
                n_image_tokens += j - i
                i = j
            else:
                j = i
                while j < len(token_ids) and token_ids[j] != self.context.image_token_id:
                    j += 1
                instructions.append(TextFill(
                    token_ids = token_ids[i:j], 
                    position_ids = list(range(i, j)), 
                    cache_ids = [list(range(n_text_tokens, n_text_tokens + j - i)) for _ in range(self.context.n_layers)], 
                    kv_cache_ids = self.text_kv_cache_ids, 
                    sample = j==len(token_ids), 
                    sample_dst= None,
                ))
                n_text_tokens += j - i
                i = j 
        curr_instruction = instructions[-1]
        instructions.append(ReAlloc(
            n_tokens = [n_text_tokens + n_image_tokens for _ in range(self.context.n_layers)],
            kv_cache_ids = self.image_kv_cache_ids, 
        ))
        instructions.append(Mov(
            src_cache_ids = [list(range(0, n_text_tokens)) for _ in range(self.context.n_layers)], 
            dst_cache_ids = [list(range(n_image_tokens, n_image_tokens + n_text_tokens)) for _ in range(self.context.n_layers)], 
            src_kv_cache_ids = self.text_kv_cache_ids, 
            dst_kv_cache_ids = self.image_kv_cache_ids, 
        ))
        instructions.append(ReAlloc(
            n_tokens = [0 for _ in range(self.context.n_layers)],
            kv_cache_ids = self.text_kv_cache_ids, 
        ))

        for i in range(self.config.max_tokens):
            instruction = TextFill(
                token_ids = None,
                position_ids = [curr_instruction.position_ids[-1] + 1], 
                cache_ids = [[curr_instruction.position_ids[-1] + 1] for layer_cache_ids in curr_instruction.cache_ids], 
                kv_cache_ids = self.image_kv_cache_ids, 
                sample = True
                )
            curr_instruction.next_instruction = instruction
            instructions.append(instruction)
            curr_instruction = instruction
        instructions.append(EmptyInstruction())
        curr_instruction.sample_dst = instructions[-1]
        return instructions, self.n_virtual_kv_caches

class FastCodeGenerator:
    def __init__(self, config: CompilerConfig, context: CompilerContext):
        self.config = config
        self.context = context
        self.n_embed_output_tokens = config.n_embed_output_tokens

    def code_generate(self, token_ids: list[int], pixel_values: Tensor) -> tuple[list[Instruction], int]:
        total_image_tokens = 0
        pruned_token_ids: list[int] = []
        for token_id in token_ids:
            if token_id == self.context.image_token_id:
                if total_image_tokens < self.n_embed_output_tokens:
                    total_image_tokens += 1
                    pruned_token_ids.append(token_id)
            else:
                pruned_token_ids.append(token_id)
        token_ids = pruned_token_ids

        instructions: list[Instruction] = []
        instructions.append(ImageEmbed(
            pixel_values = pixel_values, 
            image_featues_dst = None, 
            token_pruning_params = {'policy': self.config.token_pruning_policy, 'n_output_tokens': self.n_embed_output_tokens}, 
        ))
        instructions.append(ImageEmbedFill(
            image_featues = None,
            token_ids = token_ids,
            position_ids = list(range(0, len(token_ids))), 
            cache_ids = [list(range(0, len(token_ids))) for _ in range(self.context.n_layers)], 
            kv_cache_ids = list(range(self.context.n_layers)), 
            sample = True, 
            sample_dst = None, 
        ))
        instructions[0].image_featues_dst = instructions[1]
        for _ in range(self.config.max_tokens):
            curr_instruction = instructions[-1]
            instructions.append(TextFill(
                token_ids = None,
                position_ids = [curr_instruction.position_ids[-1] + 1], 
                cache_ids = [[layer_cache_ids[-1] + 1] for layer_cache_ids in curr_instruction.cache_ids], 
                kv_cache_ids = curr_instruction.kv_cache_ids,
                sample = True, 
                sample_dst= None,
            ))
        instructions.append(EmptyInstruction())
        for i in range(len(instructions) - 1):
            instructions[i].sample_dst = instructions[i + 1]
        
        n_virtual_kv_caches = self.context.n_layers
        return instructions, n_virtual_kv_caches