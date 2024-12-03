from tqdm import tqdm
import time
import random
from itertools import accumulate
from transformers import AutoTokenizer, AutoProcessor
from PIL import Image
from dataclasses import dataclass
import torch
from torch import Tensor
from typing import Literal
from dxz.engine.isa import Instruction, Fill, TextFill, ImageFill, Mov, ReAlloc, EmptyInstruction, ImageEmbedFill, ImageEmbed
from dxz.model.downloader import download_hf_model
from dxz.model.llava import LlavaForConditionalGeneration
from dxz.model.parameters import AttentionParameters, ModelParameters
from dxz.sequence.sequence import Sequence
from dxz.memory.compiler import CompilerConfig, Compiler
from dxz.memory.virtual_kv_cache import VirtualKVCache, MemoryManagementUnit, MemoryConfig
from dxz.utils.profiler import profile

@dataclass
class EngineConfig:
    model_name  : str                                                   = "llava-hf/llava-1.5-7b-hf"  # the repository name of huggingface
    dtype       : torch.dtype                                           = torch.half                  # 
    device      : torch.device                                          = torch.device('cuda:0')      #
    batch_policy: Literal['nobatch', 'requestlevel', 'continuousbatch'] = 'nobatch'                   #
    num_blocks  : int                                                   = 16
    block_size  : int                                                   = 16                          # kvcache block size
    token_prunning_policy: Literal['vanilla', 'mchunkprefill', 'random', 'streamingllm', 'block_prefill'] = "streamingllm"
    # streamingLLM params
    window_size: int = 16
    attention_sink_size: int = 4 

class SequenceScheduler:
    def __init__(self, batch_policy: str):
        self.waiting: list[Sequence] = []
        self.running: list[Sequence] = []
        self.finished: list[Sequence] = []
        self.batch_policy = batch_policy
    
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

    def step(self) -> list[tuple[Sequence, Instruction]]:
        if self.batch_policy == 'nobatch':
            if len(self.running) == 0:
                if len(self.waiting) != 0:
                    self.running.append(self.waiting.pop())
                else:
                    return []
            running = self.running
            self.running = []
            return [(running[0], running[0].next_instruction())]
        elif self.batch_policy == 'requestlevel':
            if len(self.running) == 0:
                while len(self.running) < 4 and len(self.waiting) != 0:
                    self.running.append(self.waiting.pop())
                if len(self.running) == 0:
                    return []
            running = self.running
            self.running = []
            return [(seq, seq.next_instruction()) for seq in running]
        elif self.batch_policy == 'continuousbatch':
            while len(self.running) < 4 and len(self.waiting) != 0:
                self.running.append(self.waiting.pop())
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
            max_tokens = 50,
            n_layers = self.model_config.text_config.num_hidden_layers,
            token_prunning_policy = self.config.token_prunning_policy, 
            # streamingLLM params
            window_size = self.config.window_size, 
            attention_sink_size = self.config.attention_sink_size
        )
        self.compiler = Compiler(self.compiler_config)
        # 4. sequence
        self.sid_allocator = 0
        self.scheduler = SequenceScheduler(config.batch_policy)

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
            self.compiler_config.max_tokens = input.get('max_tokens', 50)
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
                eos_token_id = input.get('eos_token_id', None), 
                max_seq_len = self.model_config.text_config.max_position_embeddings, 
            )
            sequence.metric.arrival_time = arrival_time
            self.sid_allocator += 1
            self.scheduler.schedule_new([sequence])

        outputs = []
        finished: list[Sequence] = []
        bar = tqdm(len(inputs))
        while len(finished) < len(inputs):
            self.step()
            f = self.scheduler.pop_finished() 
            finished += f
            bar.update(len(f))

        finished = sorted(finished, key=lambda seq: seq.sid)

        for sequence in finished:
            outputs.append({
                'text' : self.tokenizer.decode(sequence.output_token_ids, skip_special_tokens=True), 
                'ttft' : sequence.metric.tokens_time[0] - arrival_time,
                'tpot' : [sequence.metric.tokens_time[i] - sequence.metric.tokens_time[i - 1] for i in range(1, len(sequence.metric.tokens_time))], 
                'latency' : sequence.metric.finished_time - sequence.metric.arrival_time
            })

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

    @torch.inference_mode()
    def step(self):
        # 1. schedule sequence
        contexts = self.scheduler.step()
        if len(contexts) == 0:
            return

        # 2. execute instructions
        fill_contexts = []
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
                with profile('embed'):
                    pixel_values = instruction.pixel_values.to(self.config.dtype).to(self.config.device)
                    model_params = ModelParameters(embed_token_pruning_params=instruction.token_pruning_params)
                    image_features = self.model.image_embed(pixel_values, model_params)
                    instruction.image_featues_dst.image_features = image_features
                continue
            raise Exception(f'unsupported instrction {type(instruction)}')
        with profile('fill'):
            self.execute_batch_fill(fill_contexts)

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
    # ['vanilla', 'mchunkprefill', 'random', 'streamingllm', 'block_prefill', 'fast']
    config = EngineConfig(
        batch_policy = 'continuousbatch', 
        token_prunning_policy = "vanilla", 
        window_size = 128, 
        attention_sink_size = 1, 
    )
    engine = Engine(config)
    batch_size = 10

    inputs = [{
        "prompt" : prompt, 
        "multi_modal_data":{
            "image": image
        },
        "max_tokens":random.randint(30, 70), 
    } for _ in range(batch_size)]

    import time
    outputs = engine.generate(inputs)

    for output in outputs:
        print(output['text'])
        # print(f"ttft: {output['ttft']}")
        # print(f"tpot: {output['tpot']}")
        print(f"latency: {output['latency']}")