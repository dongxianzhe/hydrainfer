from tqdm import tqdm
import time
import random
from itertools import accumulate
from transformers import AutoTokenizer, AutoProcessor
from PIL import Image
from dataclasses import dataclass, field
import torch
from torch import Tensor
from typing import Literal
from dxz.engine.isa import Instruction, Fill, TextFill, ImageFill, Mov, ReAlloc, EmptyInstruction, ImageEmbedFill, ImageEmbed
from dxz.model.downloader import download_hf_model
from dxz.model.llava import LlavaForConditionalGeneration
from dxz.model.parameters import AttentionParameters, ModelParameters
from dxz.sequence.sequence import Sequence
from dxz.memory.compiler import CompilerConfig, Compiler, CompilerContext, CompileParameters
from dxz.memory.virtual_kv_cache import VirtualKVCache, MemoryManagementUnit, MemoryConfig, MemoryContext
from queue import Queue

@dataclass
class SchedulerConfig:
    batch_policy: Literal['nobatch', 'requestlevel', 'continuousbatch'] = 'continuousbatch'
    max_running_sequences: int = 10
    max_batch_tokens: int = 1024

@dataclass
class EngineConfig:
    model_name  : str          = "llava-hf/llava-1.5-7b-hf" 
    dtype       : torch.dtype  = torch.half 
    device      : torch.device = torch.device('cuda:0') 
    memory_config    : MemoryConfig    = field(default_factory=MemoryConfig)
    compiler_config  : CompilerConfig  = field(default_factory=CompilerConfig)
    scheduler_config : SchedulerConfig = field(default_factory=SchedulerConfig)
    batch_image_embed: bool = True

@dataclass
class GenerateOutput:
    input_len: int
    text : str
    ttft : float
    tpot : list[float]
    latency : float

class SequenceScheduler:
    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.waiting = Queue()
        self.running: list[Sequence] = []
        self.finished: list[Sequence] = []
    
    def schedule_new(self, sequences: list[Sequence]):
        for sequence in sequences:
            self.waiting.put(sequence)
    
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

    def step(self) -> list[tuple[Sequence, Instruction]]:
        if self.config.batch_policy == 'nobatch':
            if len(self.running) == 0:
                if not self.waiting.empty():
                    sequence = self.waiting.get()
                    sequence.metric.first_schedule_time = time.perf_counter()
                    self.running.append(sequence)
                else:
                    return []
            running = self.running
            self.running = []
            return [(running[0], running[0].next_instruction())]
        elif self.config.batch_policy == 'requestlevel':
            if len(self.running) == 0:
                while len(self.running) < self.config.max_running_sequences and not self.waiting.empty():
                    sequence = self.waiting.get()
                    sequence.metric.first_schedule_time = time.perf_counter()
                    self.running.append(sequence)
                if len(self.running) == 0:
                    return []
            running = self.running
            self.running = []
            return [(seq, seq.next_instruction()) for seq in running]
        elif self.config.batch_policy == 'continuousbatch':
            while len(self.running) < self.config.max_running_sequences and not self.waiting.empty():
                sequence = self.waiting.get()
                sequence.metric.first_schedule_time = time.perf_counter()
                self.running.append(sequence)
            if len(self.running) == 0:
                return []
            running = self.running
            self.running = []
            return [(seq, seq.next_instruction()) for seq in running]

    def __repr__(self):
        return f'{len(self.waiting)} {len(self.running)} {len(self.finished)}'

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
        self.memory_context = MemoryContext(
            head_size = self.model_config.text_config.head_dim, 
            num_kv_heads=self.model_config.text_config.num_key_value_heads, 
            dtype = self.config.dtype, 
            device = self.config.device, 
        )
        self.mmu = MemoryManagementUnit(
            config = self.config.memory_config, 
            context = self.memory_context, 
            )
        # 3. compiler
        self.compiler_context = CompilerContext(
            tokenizer = self.tokenizer, 
            processor = self.processor, 
            image_token_id = self.model_config.image_token_index, 
            num_image_tokens = self.model_config.image_seq_length, 
            n_layers = self.model_config.text_config.num_hidden_layers,
        )
        self.compiler = Compiler(
            config = self.config.compiler_config, 
            context = self.compiler_context, 
        )
        # 4. sequence
        self.sid_allocator = 0
        self.scheduler = SequenceScheduler(self.config.scheduler_config)

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
        arrival_time = time.perf_counter()
        for input in inputs:
            static_info = self.compiler.compile(
                prompt = input['prompt'], 
                images = [input['multi_modal_data']['image']], 
                compile_params = CompileParameters(max_tokens = input.get('max_tokens', None))
                )
            sequence = Sequence(
                static_info=static_info, 
                sid = self.sid_allocator, 
                instructions = static_info.instructions, 
                virtual_kv_caches = self.mmu.allocate_virtual_kv_caches(static_info.n_virtual_kv_caches), 
                max_tokens = input.get('max_tokens', 50), 
                eos_token_id = input.get('eos_token_id', None), 
                max_seq_len = self.model_config.text_config.max_position_embeddings, 
            )
            sequence.metric.arrival_time = arrival_time
            self.sid_allocator += 1
            self.scheduler.schedule_new([sequence])

        outputs = []
        finished: list[Sequence] = []
        bar = tqdm(range(len(inputs)))
        while len(finished) < len(inputs):
            self.step()
            f = self.scheduler.pop_finished() 
            finished += f
            bar.update(len(f))

        finished = sorted(finished, key=lambda seq: seq.sid)

        for sequence in finished:
            outputs.append(GenerateOutput(
                input_len = sequence.static_info.n_prompt_tokens, 
                text = self.tokenizer.decode(sequence.output_token_ids, skip_special_tokens=True), 
                ttft = sequence.metric.tokens_time[0] - sequence.metric.arrival_time,
                tpot = [sequence.metric.tokens_time[i] - sequence.metric.tokens_time[i - 1] for i in range(1, len(sequence.metric.tokens_time))], 
                latency = sequence.metric.finished_time - sequence.metric.arrival_time
            ))

        return outputs
    
    def execute_batch_fill(self, contexts: list[tuple[Sequence, Instruction]]):
        if len(contexts) == 0:
            return

        # 1. prepare input
        num_sequences     : int
        token_ids         : list[int] = []
        position_ids      : list[int] = []
        q_seq_lens        : list[int] = []
        selected_token_ids: list[int] = []
        pixel_values      : list[Tensor] = []
        image_featues     : list[Tensor] = []
        kv_seq_lens       : list[list[int]] = [[] for _ in range(self.model_config.text_config.num_hidden_layers)]
        block_tables      : list[list[int]] = [[] for _ in range(self.model_config.text_config.num_hidden_layers)]
        blocks_lens       : list[list[int]] = [[] for _ in range(self.model_config.text_config.num_hidden_layers)]
        new_cache_slots   : list[list[int]] = [[] for _ in range(self.model_config.text_config.num_hidden_layers)]
        has_image_fill: bool = False
        has_image_embed_fill: bool = False

        num_sequences = len(contexts)
        for sequence, instruction in contexts:
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
                pixel_values.append(instruction.pixel_values)
                instruction.pixel_values = None
                has_image_fill = True
            if isinstance(instruction, ImageEmbedFill):
                image_featues.append(instruction.image_features)
                instruction.image_features = None
                has_image_embed_fill = True

        if has_image_fill and has_image_embed_fill:
            raise Exception('not support pixel value and image embed batch')

        q_max_seq_len : int = max(q_seq_lens)

        # 2. prepare tensor data
        q_cu_seq_lens: Tensor = torch.tensor(list(accumulate(q_seq_lens, initial=0)), dtype=torch.int, device=self.config.device)
        kv_cu_seq_lens = torch.tensor([list(accumulate(kv_seq_lens[layer_id], initial=0)) for layer_id in range(self.model_config.text_config.num_hidden_layers)], dtype=torch.int, device=self.config.device)
        new_cache_slots = torch.tensor(new_cache_slots, dtype=torch.int, device=self.config.device)
        block_tables = torch.tensor(block_tables, dtype=torch.int, device=self.config.device)
        cu_blocks_lens = torch.tensor([list(accumulate(blocks_lens[layer_id], initial=0)) for layer_id in range(self.model_config.text_config.num_hidden_layers)], dtype=torch.int, device=self.config.device)
        model_params = ModelParameters(
            attention_params = [AttentionParameters(
                kv_cache=self.mmu.kv_cache,
                q_cu_seq_lens = q_cu_seq_lens, 
                kv_cu_seq_lens = kv_cu_seq_lens[layer_id], 
                new_cache_slots = new_cache_slots[layer_id], 
                block_tables = block_tables[layer_id], 
                cu_blocks_lens = cu_blocks_lens[layer_id], 
                num_sequences = num_sequences, 
                all_sequences_decode = False, 
                q_max_seq_len = q_max_seq_len,
                kv_max_seq_len = max(kv_seq_lens[layer_id])
            )for layer_id in range(self.model_config.text_config.num_hidden_layers)]
        )
        input_ids = torch.tensor(token_ids, dtype=torch.int, device=self.config.device)
        position_ids = torch.tensor(position_ids, dtype=torch.int, device=self.config.device)
        if len(pixel_values):
            pixel_values = torch.cat(pixel_values, dim=0).to(dtype=self.config.dtype, device=self.config.device)
        else:
            pixel_values = None
        if len(image_featues):
            image_featues = torch.cat(image_featues, dim=0).to(dtype=self.config.dtype, device=self.config.device)
        else:
            image_featues = None
            
        logits = self.model_runner(input_ids, pixel_values, image_featues, position_ids, model_params)
        if len(selected_token_ids) > 0:
            t = time.perf_counter()
            sample_token_ids = torch.argmax(logits[selected_token_ids, :], dim=-1, keepdim=False).tolist()
            i = 0
            for sequence, instruction in contexts:
                if (isinstance(instruction, Fill)) and instruction.sample:
                    next_token_id = sample_token_ids[i]
                    instruction.sample_dst.token_ids = [next_token_id]
                    sequence.output_token_ids.append(next_token_id)
                    sequence.metric.tokens_time.append(t)
                    i += 1

    def execute_mov(self, context: tuple[Sequence, Instruction]):
        sequence, instruction = context
        src_slot_ids: list[int] = []
        for src_cache_ids, src_kv_cache_id in zip(instruction.src_cache_ids, instruction.src_kv_cache_ids):
            block_table = sequence.virtual_kv_caches[src_kv_cache_id].block_tables
            src_slot_ids += self.mmu.v2p(src_cache_ids, block_table)
        dst_slot_ids: list[int] = []
        for dst_cache_ids, dst_kv_cache_id in zip(instruction.dst_cache_ids, instruction.dst_kv_cache_ids):
            block_table = sequence.virtual_kv_caches[dst_kv_cache_id].block_tables
            dst_slot_ids += self.mmu.v2p(dst_cache_ids, block_table)
        assert len(src_slot_ids) == len(dst_slot_ids), f'{len(src_slot_ids)} {len(dst_slot_ids)}'
        src_slot_ids = torch.tensor(src_slot_ids, dtype=torch.int, device=self.config.device)
        dst_slot_ids = torch.tensor(dst_slot_ids, dtype=torch.int, device=self.config.device)
        self.mmu.move_physical_kv_caches(src_slot_ids, dst_slot_ids)


    def execute_realloc(self, context: tuple[Sequence, Instruction]):
        sequence, instruction = context
        for n_token, vid in zip(instruction.n_tokens, instruction.kv_cache_ids):
            sequence.virtual_kv_caches[vid].realloc(n_token)

    def execute_batch_image_embed(self, contexts: list[tuple[Sequence, Instruction]]):
        if len(contexts) == 0:
            return

        n_images: list[int] = []
        batch_pixel_values: list[Tensor] = []
        for sequence, instruction in contexts:
            pixel_values = instruction.pixel_values.to(self.config.dtype).to(self.config.device) # (n_images, n_channels, width, height)
            batch_pixel_values.append(pixel_values)
            n_images.append(pixel_values.shape[0])
        pixel_values = torch.cat(batch_pixel_values, dim=0) 

        model_params = ModelParameters(embed_token_pruning_params=instruction.token_pruning_params)
        image_features = self.model.image_embed(pixel_values, model_params) # (batch_size, n_tokens, vision_hidden_size)

        left = 0
        for i, (sequence, instruction) in enumerate(contexts):
            right = left + n_images[i]
            instruction.image_featues_dst.image_features = image_features[left: right, :, :]
            left += n_images[i]

    def execute_image_embed(self, context: tuple[Sequence, Instruction]):
        sequence, instruction = context
        pixel_values = instruction.pixel_values.to(self.config.dtype).to(self.config.device)
        model_params = ModelParameters(embed_token_pruning_params=instruction.token_pruning_params)
        image_features = self.model.image_embed(pixel_values, model_params)
        instruction.image_featues_dst.image_features = image_features

    @torch.inference_mode()
    def step(self):
        # 1. schedule sequence
        contexts = self.scheduler.step()
        if len(contexts) == 0:
            return

        # 2. execute instructions
        fill_contexts = []
        image_embed_contexts = []
        for context in contexts:
            sequence, instruction = context
            if isinstance(instruction, Fill):
                fill_contexts.append(context)
                continue
            if isinstance(instruction, Mov):
                self.execute_mov(context)
                continue
            if isinstance(instruction, ReAlloc):
                self.execute_realloc(context)
                continue
            if isinstance(instruction, EmptyInstruction):
                continue
            if isinstance(instruction, ImageEmbed):
                image_embed_contexts.append(context)
                continue
            raise Exception(f'unsupported instrction {type(instruction)}')
        self.execute_batch_fill(fill_contexts)

        if self.config.batch_image_embed:
            self.execute_batch_image_embed(image_embed_contexts)
        else:
            for context in image_embed_contexts:
                self.execute_image_embed(context)

        # 3. scheduler sequence
        t = time.perf_counter()
        for sequence, _ in contexts:
            if sequence.is_finished():
                sequence.metric.finished_time = t
                self.scheduler.schedule_finished([sequence])
            else:
                self.scheduler.schedule_unfinished([sequence])

if __name__ == '__main__':
    image_path = f'/home/xzd/projects/dxz/benchmark/dataset/cherry_blossom.jpg'
    image = Image.open(image_path)
    question = "What is the content of this image?"
    prompt = f"USER: <image>\n{question}\nASSISTANT:"

    # ['nobatch', 'requestlevel', 'continuousbatch']
    # ['random', 'streamingllm']
    config = EngineConfig(
        model_name = "llava-hf/llava-1.5-7b-hf", 
        dtype = torch.half, 
        device = torch.device('cuda:0'), 
        memory_config=MemoryConfig(
            num_blocks = 20000, 
            block_size = 16, 
        ), 
        scheduler_config=SchedulerConfig(
            batch_policy = 'continuousbatch', 
            max_running_sequences = 10, 
        ), 
        compiler_config=CompilerConfig(
            max_tokens = 64, 
            kv_cache_eviction_policy = None, 
            window_size = 28, 
            attention_sink_size = 4, 
            token_pruning_policy = None, 
            n_embed_output_tokens = 64, 
        ), 
        batch_image_embed = True, 
    )
    engine = Engine(config)
    batch_size = 10
    inputs = [{
        "prompt" : prompt, 
        "multi_modal_data":{
            "image": image
        },
        # "max_tokens":0, 
        # "max_tokens":random.randint(30, 70), 
        "max_tokens": i * 10, 
    } for i in range(batch_size)]

    import time
    start = time.perf_counter()

    outputs = engine.generate(inputs)

    end = time.perf_counter()
    duration = end - start
    print(f'duration {duration}')
    for output in outputs:
        print(output.text)