from itertools import accumulate
from transformers import AutoTokenizer, AutoProcessor
from PIL import Image
from dataclasses import dataclass
import torch
from torch import Tensor
from typing import Literal
from dxz.engine.isa import Instruction, Fill, TextFill, ImageFill, Mov, ReAlloc
from dxz.model.downloader import download_hf_model
from dxz.model.llava import LlavaForConditionalGeneration
from dxz.model.parameters import InputParameters
from dxz.sequence.sequence import Sequence
from dxz.memory.compiler import CompilerConfig, Compiler, DecodeParams
from dxz.memory.virtual_kv_cache import VirtualKVCache, MemoryManagementUnit, MemoryConfig

@dataclass
class EngineConfig:
    model_name  : str                                                   = "llava-hf/llava-1.5-7b-hf"  # the repository name of huggingface
    dtype       : torch.dtype                                           = torch.half                  # 
    device      : torch.device                                          = torch.device('cuda:0')      #
    batch_policy: Literal['nobatch', 'requestlevel', 'continuousbatch'] = 'nobatch'                   #
    num_blocks  : int                                                   = 16
    block_size  : int                                                   = 16                          # kvcache block size

class SequenceScheduler:
    def __init__(self):
        self.waiting: list[Sequence] = []
        self.running: list[Sequence] = []
        self.finished: list[Sequence] = []
    
    def schedule_new(self, sequences: list[Sequence]):
        self.waiting += sequences
    
    def schedule_running(self, sequences: list[Sequence]):
        self.running += sequences

    def schedule_unfinished(self, sequences: list[Sequence]):
        self.running += sequences
    
    def schedule_finished(self, sequences: list[Sequence]):
        self.finished += sequences

    def pop_finished(self) -> list[Sequence]:
        finished = self.finished
        self.finished = []
        return finished

    def step(self) -> tuple[list[Sequence], list[Instruction]]:
        if len(self.running) == 0:
            if len(self.waiting) != 0:
                self.running.append(self.waiting.pop())
            else:
                return [], []
        running = self.running
        self.running = []
        return running, [running[0].next_instruction()]

    def __repr__(self):
        return f'{len(self.waiting)} {len(self.running)} {len(self.finished)}'

from tqdm import tqdm
class Engine:
    def __init__(self, config: EngineConfig):
        self.config = config
        # 1. model
        model_path = download_hf_model(repo_id=config.model_name)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer
        self.model = LlavaForConditionalGeneration.from_safetensor(model_path, self.config.dtype, self.config.device)
        self.model_config = self.model.config
        self.model_runner = self.model
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
        self.scheduler = SequenceScheduler()

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
            static_info = self.compiler.compile(
                prompt = input['prompt'], 
                images = [input['multi_modal_data']['image']], 
                )
            sequence = Sequence(
                static_info=static_info, 
                sid = self.sid_allocator, 
                instructions = static_info.instructions, 
                virtual_kv_caches = self.mmu.allocate_virtual_kv_caches(static_info.n_virtual_kv_caches), 
                max_tokens = input.get('max_tokens', 50), 
                eos_token_id = self.tokenizer.eos_token_id, 
                max_seq_len = self.model_config.text_config.max_position_embeddings, 
            )
            self.sid_allocator += 1
            self.scheduler.schedule_new([sequence])

        output_texts: list[str] = []
        finished: list[Sequence] = []
        while len(finished) < len(inputs):
            self.step()
            finished += self.scheduler.pop_finished()

        for sequence in finished:
            output_texts.append(self.tokenizer.decode(sequence.output_token_ids, skip_special_tokens=True))

        return output_texts

    @torch.inference_mode()
    def step(self):
        # 1. scheduler sequence
        # 2. batch instruction
        sequences, instructions = self.scheduler.step()
        assert len(sequences) == len(instructions)
        if len(instructions) == 0:
            return 

        # 3. interpret instruction
        token_ids         : list[int] = []
        position_ids      : list[int] = []
        q_seq_lens        : list[int] = []
        selected_token_ids: list[int] = []

        images            : list[Tensor] = []

        kv_seq_lens       : list[list[int]] = [[] for _ in range(self.model_config.text_config.num_hidden_layers)]
        block_tables      : list[list[int]] = [[] for _ in range(self.model_config.text_config.num_hidden_layers)]
        blocks_lens       : list[list[int]] = [[] for _ in range(self.model_config.text_config.num_hidden_layers)]
        new_cache_slots   : list[list[int]] = [[] for _ in range(self.model_config.text_config.num_hidden_layers)]
        for sequence, instruction in zip(sequences, instructions):
            if isinstance(instruction, Fill):
                token_ids += instruction.token_ids
                if instruction.sample:
                    selected_token_ids.append(len(token_ids) - 1)
                position_ids += instruction.position_ids
                q_seq_lens.append(len(instruction.token_ids))
                for layer_id, (vcids, vid) in enumerate(zip(instruction.cache_ids, instruction.kv_cache_ids)):
                    sequence.virtual_kv_caches[vid].set(vcids)
                    slot_ids = self.mmu.v2p(vcids, sequence.virtual_kv_caches[vid].block_tables)
                    new_cache_slots[layer_id] += slot_ids
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
        logits = self.model_runner(input_ids, pixel_values, position_ids, self.mmu.kv_cache, input_params)
        # 4. sample
        if len(selected_token_ids) > 0:
            sample_token_ids = torch.argmax(logits[selected_token_ids, :], dim=-1, keepdim=False).tolist()
            i = 0
            for sequence, instruction in zip(sequences, instructions):
                if (isinstance(instruction, Fill)) and instruction.sample:
                    next_token_id = sample_token_ids[i]
                    i += 1
                    next_instruction = self.compiler.interpret_next_instruction(DecodeParams(n_prompt_tokens=sequence.static_info.n_prompt_tokens, curr_instruction=instruction, next_token_id=next_token_id))
                    sequence.append_instruction(next_instruction)
                    sequence.output_token_ids.append(next_token_id)
        # 5. scheduler sequence
        for sequence in sequences:
            if sequence.is_finished():
                self.scheduler.schedule_finished([sequence])
            else:
                self.scheduler.schedule_unfinished([sequence])

if __name__ == '__main__':
    
    image_path = f'/home/xzd/projects/dxz/benchmark/dataset/cherry_blossom.jpg'
    image = Image.open(image_path)
    question = "What is the content of this image?"
    prompt = f"USER: <image>\n{question}\nASSISTANT:"

    config = EngineConfig()
    engine = Engine(config)

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
    print(f'total time {end - start: 3}')