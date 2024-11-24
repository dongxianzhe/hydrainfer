from itertools import accumulate
import torch
from torch import Tensor
from typing import Literal
from dxz.model.parameters import InputParameters

from dxz.memory.kv_cache import KVCache
from dxz.memory.block_allocator import BlockAllocator

from dataclasses import dataclass

@dataclass
class MemoryConfig:
    num_blocks: int
    block_size: int
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    device: torch.device
     

class MemoryManagementUnit:
    """_summary_
        MemoryManagementUnit manage 
        1. cpu memory
        2. gpu memory
        3. vitual kv cache creation
    """
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.kv_cache = KVCache(
            config.num_blocks, 
            config.block_size, 
            config.num_kv_heads, 
            config.head_size, 
            config.dtype, 
            config.device
        )
        self.kv_cache_allocator = BlockAllocator(config.num_blocks)

    def allocate(self, n_blocks: int) -> list[int]:
        return self.kv_cache_allocator.allocate(n_blocks)
    
    def free(self, blocks: list[int]) -> bool:
        return self.kv_cache_allocator.free(blocks)


class VirtualKVCache:
    def __init__(self, mmu: MemoryManagementUnit):
        self.config: MemoryConfig = mmu.config
        self.block_size = self.config.block_size
        self.mmu = mmu

        self.n_kv_cache_tokens: int = 0
        self.block_tables: list[int] = []

    def set(self, virtual_cache_ids: list[int]) -> bool:
        """
            fill kv cache of cache_ids with tokens, cache_ids (n_tokens, )
            will allocate memory automatic if necessary
            return True means set success
            False means set failed due to not enough memory
        """
        # 1. try to allocate memory if block is not enough
        n_tokens = max(virtual_cache_ids) + 1
        n_blocks = (n_tokens + self.block_size - 1) // self.block_size
        if len(self.block_tables) < n_blocks:
            self.block_tables += self.mmu.allocate(n_blocks - len(self.block_tables))
        if len(self.block_tables) < n_blocks:
            raise Exception('not enough kv cache')
            return False
        # 2. set vitual kv cache
        self.n_kv_cache_tokens = max(self.n_kv_cache_tokens, n_tokens)
        return True

    def free_blocks(self, virtual_block_ids: list[int]) -> bool:
        for virtual_block_id in sorted(virtual_block_ids, reverse=True):
            physical_block_id = self.block_tables[virtual_block_id]
            self.mmu.free([physical_block_id])
            del self.block_tables[virtual_block_id]
        return True

class Instruction:
    pass

class Fill(Instruction):
    def __init__(self, token_ids: list[int], position_ids: list[int], cache_ids: list[list[int]], kv_cache_ids: list[int], sample: bool):
        # cache_ids (n_layers, n_tokens)
        # kv_caches (n_layers, )
        super().__init__()
        self.token_ids = token_ids
        self.position_ids = position_ids
        self.cache_ids = cache_ids
        self.kv_cache_ids = kv_cache_ids
        self.sample = sample

    def __repr__(self):
        return f"Fill {self.token_ids} {self.position_ids} {self.kv_cache_ids}"

class ImageFill(Instruction):
    def __init__(self, images: list[Tensor], token_ids: list[int], position_ids: list[int], cache_ids: list[list[int]], kv_cache_ids: list[int], sample: bool):
        super().__init__()
        self.images    = images
        self.token_ids = token_ids
        self.position_ids = position_ids
        self.cache_ids = cache_ids
        self.kv_cache_ids = kv_cache_ids
        self.sample = sample

    def __repr__(self):
        return f"ImageFill {self.token_ids[:3]}...{self.token_ids[-3:]} {self.position_ids[:3]}...{self.position_ids[-3:]} {self.kv_cache_ids}"

class Mov(Instruction):
    def __init__(self, src_cache_ids: list[list[int]], dst_cache_ids: list[list[int]], src_kv_cache_ids: list[int], dst_kv_cache_ids: list[int]):
        super().__init__()
        self.src_cache_ids = src_cache_ids
        self.dst_cache_ids = dst_cache_ids
        self.src_kv_cache_ids = src_kv_cache_ids
        self.dst_kv_cache_ids = dst_kv_cache_ids

class Merge(Instruction):
    def __init__(self, kv_cache1_ids: list[int], kv_cache2_ids: list[int]):
        super().__init__()
        self.kv_cache1_ids = kv_cache1_ids
        self.kv_cache2_ids = kv_cache2_ids

class Sequence:
    def __init__(self, sid: int, instructions: list[Instruction], virtual_kv_caches: list[VirtualKVCache], max_tokens: int, eos_token_id: int, max_seq_len: int):
        self.sid: int = sid
        self.pc: int = 0
        self.instructions: list[Instruction] = instructions
        self.n_virtual_kv_caches: int
        self.virtual_kv_caches: list[VirtualKVCache] = virtual_kv_caches

        self.output_token_ids: list[int] = []
        self.max_tokens = max_tokens
        self.eos_token_id = eos_token_id
        self.max_seq_len = max_seq_len
        self.append_instruction_count = 0

    def next_instruction(self) -> Instruction:
        inst = self.instructions[self.pc]
        self.pc += 1
        return inst

    def append_instruction(self, instruction: Fill):
        assert isinstance(instruction, Fill) and len(instruction.token_ids) == 1
        if instruction.token_ids[-1] != self.eos_token_id and len(self.output_token_ids) < self.max_tokens:
            self.instructions.append(instruction)
            self.append_instruction_count += 1

    def is_finished(self) -> bool:
        return self.pc >= len(self.instructions)

@dataclass
class NewEngineConfig:
    model_name  : str                                                   = "llava-hf/llava-1.5-7b-hf"  # the repository name of huggingface
    dtype       : torch.dtype                                           = torch.half                  # 
    device      : torch.device                                          = torch.device('cuda:0')      #
    batch_policy: Literal['nobatch', 'requestlevel', 'continuousbatch'] = 'nobatch'                   #
    num_blocks  : int                                                   = 16
    block_size  : int                                                   = 16                          # kvcache block size

from dxz.model.downloader import download_hf_model
from transformers import AutoTokenizer, AutoProcessor
from dxz.model.llava import LlavaForConditionalGeneration

class Context:
    pass

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
        print('----------------------------compile info--------------------------------')
        print(f'len(token_ids) {len(token_ids)}')
        for instruction in instructions:
            print(f"{instruction}")
        print('------------------------------------------------------------------------')
        return instructions, n_virtual_kv_caches

    def interpret_next_instruction(self, instruction: Instruction, token_id: int) -> Instruction:
        instruction = Fill(
            token_ids = [token_id],
            
            position_ids = [instruction.position_ids[-1] + 1], 
            cache_ids = [cache_id + 1 for cache_id in instruction.cache_ids], 
            kv_cache_ids = instruction.kv_cache_ids,
            sample = True
            )
        return instruction

class NoBatchPolicy:
    def __init__(self):
        pass

    def batch(self, sequences: list[Sequence]) -> tuple[list[Sequence], list[Instruction]]:
        assert len(sequences) == 1
        return sequences, [sequences[0].next_instruction()]

from tqdm import tqdm
class NewEngine:
    def __init__(self, config: NewEngineConfig):
        self.config = config
        # 1. model
        model_path = download_hf_model(repo_id=config.model_name)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer
        self.model = LlavaForConditionalGeneration.from_safetensor(model_path, self.config.dtype, self.config.device)
        self.model_config = self.model.config
        self.model_runner = self.model
        print(f'model[info] {config.model_name} processor {self.processor} tokenizer {type(self.tokenizer)}')
        # 2. memory
        self.memory_config = MemoryConfig(
            num_blocks = 10000, 
            block_size = 16, 
            num_kv_heads = self.model_config.text_config.num_key_value_heads, 
            head_size = self.model_config.text_config.head_dim, 
            dtype = self.config.dtype, 
            device = self.config.device, 
        )
        self.mmu = MemoryManagementUnit(self.memory_config)
        # 3. compiler
        self.compiler_config = CompilerConfig(
            tokenizer = self.tokenizer, 
            processor = self.processor, 
            image_token_id = self.model_config.image_token_index, 
            num_image_tokens = 576, 
            n_layers = self.model_config.text_config.num_hidden_layers,
        )
        self.compiler = Compiler(self.compiler_config)
        # 4. sequence
        self.sid_allocator = 0
        self.waiting: list[Sequence] = []
        self.running: list[Sequence] = []
        self.finished: list[Sequence] = []
        # 5. batch policy
        self.batch_policy = NoBatchPolicy()

    def generate(self, inputs):
        """ inputs example
        [{
            "prompt" : prompt, 
            "multi_modal_data":{
                "image": image
            },
            "max_tokens": 50,
        }, ...]
        """
        for input in inputs:
            instructions, n_virtual_kv_caches = self.compiler.compile(
                prompt = input['prompt'], 
                images = [input['multi_modal_data']['image']], 
                )
            sequence = Sequence(
                sid = self.sid_allocator, 
                instructions = instructions, 
                virtual_kv_caches = [VirtualKVCache(self.mmu) for _ in range(n_virtual_kv_caches)], 
                max_tokens = input.get('max_tokens', 50), 
                eos_token_id = self.tokenizer.eos_token_id, 
                max_seq_len = self.model_config.text_config.max_position_embeddings, 
            )
            self.sid_allocator += 1
            self.waiting.append(sequence)

        output_texts: list[str] = []
        while len(self.finished) < len(inputs):
            self.step()
            output_texts.append(self.tokenizer.decode(sequence.output_token_ids, skip_special_tokens=True))
        return output_texts

    @torch.inference_mode()
    def step(self):
        # 1. scheduler sequence
        if len(self.running) == 0:
            if len(self.waiting) != 0:
                self.running.append(self.waiting.pop())
            return
        # 2. batch instruction
        sequences    : list[Sequence]
        instructions : list[Instruction]
        sequences, instructions = self.batch_policy.batch(self.running)

        token_ids         : list[int] = []
        position_ids         : list[int] = []
        q_seq_lens        : list[int] = []
        selected_token_ids: list[int] = []

        images            : list[Tensor] = []

        kv_seq_lens       : list[list[int]] = [[] for _ in range(self.model_config.text_config.num_hidden_layers)]
        block_tables      : list[list[int]] = [[] for _ in range(self.model_config.text_config.num_hidden_layers)]
        blocks_lens       : list[list[int]] = [[] for _ in range(self.model_config.text_config.num_hidden_layers)]
        new_cache_slots   : list[list[int]] = [[] for _ in range(self.model_config.text_config.num_hidden_layers)]
        for sequence, instruction in zip(sequences, instructions):
            if isinstance(instruction, Fill) or isinstance(instruction, ImageFill):
                token_ids += instruction.token_ids
                if instruction.sample:
                    selected_token_ids.append(len(token_ids) - 1)
                position_ids += instruction.position_ids
                q_seq_lens.append(len(instruction.token_ids))
                for layer_id, (vcids, vid) in enumerate(zip(instruction.cache_ids, instruction.kv_cache_ids)):
                    sequence.virtual_kv_caches[vid].set(vcids)
                    print(f'sequence.virtual_kv_caches[vid].n_kv_cache_tokens {sequence.virtual_kv_caches[vid].n_kv_cache_tokens}')
                    for vcid in vcids:
                        block_id = vcid // self.memory_config.block_size
                        block_offset = vcid % self.memory_config.block_size
                        slot_id = sequence.virtual_kv_caches[vid].block_tables[block_id] * self.memory_config.block_size + block_offset
                        new_cache_slots[layer_id].append(slot_id)
                    kv_seq_lens[layer_id].append(sequence.virtual_kv_caches[vid].n_kv_cache_tokens)
                    block_tables[layer_id] += sequence.virtual_kv_caches[vid].block_tables
                    blocks_lens[layer_id].append(len(sequence.virtual_kv_caches[vid].block_tables))
                if isinstance(instruction, ImageFill):
                    images += instruction.images
                continue
            raise Exception(f'instruction {type(instruction)} not implemented')

        layer_input_params: list[InputParameters] = []
        q_max_seq_len = max(q_seq_lens)
        q_cu_seq_lens = torch.tensor(list(accumulate([0] + q_seq_lens)), dtype=torch.int, device=self.config.device)
        for layer_id in range(self.model_config.text_config.num_hidden_layers):
            layer_input_params.append(InputParameters(
                num_sequences = len(sequences), 
                q_cu_seq_lens = q_cu_seq_lens, 
                kv_cu_seq_lens = torch.tensor(list(accumulate([0] + kv_seq_lens[layer_id])), dtype=torch.int, device=self.config.device), 
                new_cache_slots = torch.tensor(new_cache_slots[layer_id], dtype=torch.int ,device=self.config.device), 
                block_tables = torch.tensor(block_tables[layer_id], dtype=torch.int, device=self.config.device), 
                cu_blocks_lens = torch.tensor(list(accumulate([0] + blocks_lens[layer_id])), dtype=torch.int, device=self.config.device), 
                q_max_seq_len = q_max_seq_len, 
                kv_max_seq_len = max(kv_seq_lens[layer_id]), 
                layer_id=layer_id, 
            ))
        input_params = InputParameters(
            layer_input_params=layer_input_params
        )
        input_ids = torch.tensor(token_ids, dtype=torch.int, device=self.config.device)
        position_ids = torch.tensor(position_ids, dtype=torch.int, device=self.config.device)
        if len(images):
            pixel_values = torch.cat(images, dim=0).to(dtype=self.config.dtype, device=self.config.device)
        else:
            pixel_values = None
        # 3. execute
        print(f'input_ids.shape   : {input_ids.shape}')
        if pixel_values is not None:
            print(f'pixel_values.shape: {pixel_values.shape}')
        print(f'position_ids.shape: {position_ids.shape}')
        print(f'input_ids[:8]     : {input_ids[:8]}')
        print(f'position_ids[:8]  : {position_ids[:8]}')
        input_params.print()
        logits = self.model_runner(input_ids, pixel_values, position_ids, self.mmu.kv_cache, input_params)
        # 4. sample
        if len(selected_token_ids) > 0:
            sample_token_ids = torch.argmax(logits[selected_token_ids, :], dim=-1, keepdim=False).tolist()
            i = 0
            for sequence, instruction in zip(sequences, instructions):
                if (isinstance(instruction, Fill) or isinstance(instruction, ImageFill)) and instruction.sample:
                    token_id = sample_token_ids[i]
                    i += 1
                    instruction = self.compiler.interpret_next_instruction(instruction=instruction, token_id=token_id)
                    sequence.append_instruction(instruction)

        # 5. scheduler sequence
        for sequence in sequences:
            if sequence.is_finished():
                self.finished.append(sequence)

if __name__ == '__main__':
    image_path = f'/home/xzd/projects/dxz/benchmark/dataset/cherry_blossom.jpg'
    image = Image.open(image_path)
    question = "What is the content of this image?"
    prompt = f"USER: <image>\n{question}\nASSISTANT:"

    config = NewEngineConfig()
    engine = NewEngine(config)

    inputs = [{
        "prompt" : prompt, 
        "multi_modal_data":{
            "image": image
        },
    }]

    import time
    start = time.perf_counter()
    outputs = engine.generate(inputs)
    end = time.perf_counter()
    print(outputs)