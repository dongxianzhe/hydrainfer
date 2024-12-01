from tqdm import tqdm
import torch
from torch import Tensor
import queue
from dataclasses import dataclass, field
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor
from typing import Literal
from dxz.model.downloader import download_hf_model
from dxz.model.llava import LlavaForConditionalGeneration
from dxz.memory.kv_cache import KVCache
from dxz.memory.block_allocator import BlockAllocator
from dxz.model.parameters import ModelParameters, AttentionParameters
from dataclasses import dataclass
from dxz.memory.block_allocator import BlockAllocator

"""this file is deprecated"""

@dataclass
class MultiModalSequence:
    prompt: str = ""
    token_ids: list[int] = field(default_factory=list)
    num_prompt_tokens: int = 0 # the number of prompt tokens
    n_kv_cache_tokens: int = 0 # the number of tokens already in kv cache
    block_table: list[int] = field(default_factory=list)
    pixel_values: Tensor = None
    # stop criterian
    max_tokens: int = 50
    eos_token_id:int = 0
    max_seq_len:int = 1024

    @property
    def is_finished(self) -> bool:
        return self.token_ids[-1] == self.eos_token_id or (len(self.token_ids) - self.num_prompt_tokens) == self.max_tokens or len(self.token_ids) > self.max_seq_len

    @property
    def is_prefill(self) -> bool:
        return self.n_kv_cache_tokens < self.num_prompt_tokens
    
    @property
    def is_prefill_but_not_new(self) -> bool:
        return self.n_kv_cache_tokens < self.num_prompt_tokens and not self.is_new

    @property
    def is_decode(self) -> bool:
        return self.n_kv_cache_tokens >= self.num_prompt_tokens
    
    @property
    def is_decode_but_not_finished(self) -> bool:
        return self.n_kv_cache_tokens >= self.num_prompt_tokens and not self.is_finished

    @property
    def is_new(self) -> bool:
        return self.n_kv_cache_tokens == 0

class NoBatchPolicy:
    def __init__(self, allocator: BlockAllocator, block_size: int):
        # this is a no-batch policy, which means we only have one sequence at a time
        self.new_queue = queue.Queue()
        self.intermediate_sequence = None # the sequence which is prefilling or decoding
        self.allocator = allocator
        self.block_size = block_size
    
    def add_new_sequence(self, sequence: MultiModalSequence):
        self.new_queue.put(sequence)

    def add_intermediate_sequences(self, batch_sequences: list[MultiModalSequence]):
        assert len(batch_sequences) == 1, "no batch policy only support batch size = 1"
        sequence = batch_sequences[0]
        if sequence.is_finished:
            self.intermediate_sequence = None
            self.allocator.free(sequence.block_table)
            sequence.block_table = []
        else:
            self.intermediate_sequence = sequence

    def batch(self) -> tuple[list[MultiModalSequence], list[int], list[Tensor]]:
        batch_sequences: list[MultiModalSequence] = []
        q_seq_lens: list[int] = []
        pixel_values: list[Tensor] = []
        # 1. select a sequence
        sequence = None
        if self.intermediate_sequence is None:
            if not self.new_queue.empty():
                sequence = self.new_queue.get()
        else:
            sequence = self.intermediate_sequence
        if sequence is None:
            return batch_sequences, q_seq_lens, pixel_values

        # 2. allocate block
        if len(sequence.block_table) * self.block_size < len(sequence.token_ids):
            n_blocks = (len(sequence.token_ids) - len(sequence.block_table) * self.block_size + self.block_size - 1) // self.block_size
            blocks = self.allocator.allocate(n_blocks)
            assert len(blocks) > 0, "No more blocks available"
            sequence.block_table += blocks

        batch_sequences.append(sequence)
        q_seq_lens.append(len(sequence.token_ids) - sequence.n_kv_cache_tokens)
        if sequence.is_new:
            pixel_values.append(sequence.pixel_values)
        return batch_sequences, q_seq_lens, pixel_values

@dataclass
class MLLMEngineConfig:
    model_name  : str                                                   = "llava-hf/llava-1.5-7b-hf"  # the repository name of huggingface
    dtype       : torch.dtype                                           = torch.half                  # 
    device      : torch.device                                          = torch.device('cuda:0')      #
    batch_policy: Literal['nobatch', 'requestlevel', 'continuousbatch'] = 'nobatch'                   #
    kvcache_size: float                                                 = 8.                          # allocate kvcache_size GB as kvcache
    block_size  : int                                                   = 16                          # kvcache block size

class MLLMEngine:
    def __init__(self, config: MLLMEngineConfig):
        self.config = config
        # 1. init model
        model_path = download_hf_model(repo_id=config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = LlavaForConditionalGeneration.from_safetensor(model_path, self.config.dtype, self.config.device)
        self.model_config = self.model.config
        self.model_runner = self.model
        print(f'using tokenizer {type(self.tokenizer)}')
        # 2. kv cache 
        sizeof_element = torch.tensor(0, dtype=config.dtype).element_size()
        self.num_blocks = int(config.kvcache_size * 1024 * 1024 * 1024 / sizeof_element / 2 / self.model_config.text_config.num_hidden_layers / self.model_config.text_config.hidden_size / config.block_size)
        self.block_size = config.block_size
        self.num_kv_heads = self.model_config.text_config.num_key_value_heads
        self.head_size = self.model_config.text_config.head_dim
        self.kv_caches = [KVCache(
            num_blocks = self.num_blocks,
            block_size = self.block_size,
            num_kv_heads = self.num_kv_heads, 
            head_size = self.head_size,
            dtype = config.dtype,
            device = config.device
            ) for _ in range(self.model_config.text_config.num_hidden_layers)]
        self.allocator = BlockAllocator(total_blocks = self.num_blocks)
        print(f'kvcaches shape {len(self.kv_caches)} {self.num_blocks} {self.block_size} {self.num_kv_heads} {self.head_size}')

        # 3. batch policy
        if config.batch_policy == 'nobatch':
            self.batch_policy = NoBatchPolicy(allocator=self.allocator, block_size=self.block_size)
        else:
            raise Exception(f'unsupportede batch policy {config.batch_policy}')
        
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
            self.add_request(input)

        pbar = tqdm(total=len(inputs))
        output_texts: list[str] = []
        while len(output_texts) < len(inputs): 
            decode_output, finished_output = self.step()
            for sequence in finished_output:
                output_texts.append(sequence.prompt + self.tokenizer.decode(sequence.token_ids[sequence.num_prompt_tokens:], skip_special_tokens=True))
            pbar.update(len(finished_output))
        pbar.close()
        return output_texts
    
    def add_request(self, input):
        # 1. tokenize prompt
        # token_ids = self.tokenizer.encode(input['prompt'])
        token_ids = self.processor(
            text=input['prompt'], 
            images=None
        )['input_ids'][0]
        pixel_values = self.processor(
            text="", 
            images=input['multi_modal_data']['image'], 
            return_tensors="pt"
        )['pixel_values']

        # 2. insert image_token_ids
        token_ids = self._insert_image_token_ids(token_ids=token_ids, image_token_id=self.model_config.image_token_index, num_token_insert=575)
        self.batch_policy.add_new_sequence(MultiModalSequence(
            prompt=input['prompt'], 
            token_ids = token_ids, 
            num_prompt_tokens = len(token_ids),
            n_kv_cache_tokens = 0, 
            pixel_values = pixel_values, 
            max_tokens = input.get('max_tokens', 50), 
            eos_token_id = self.tokenizer.eos_token_id, 
            max_seq_len = self.model_config.text_config.max_position_embeddings
        ))
    
    @torch.inference_mode()
    def step(self):
        decode_output: list[MultiModalSequence] = []
        finished_output: list[MultiModalSequence] = []
        # 1. batch
        batch_sequences, q_seq_lens, pixel_values = self.batch_policy.batch()
        if len(batch_sequences) == 0:
            return decode_output, finished_output
        # 2. prepare input
        token_ids         : list[int] = []
        positions         : list[int] = []
        q_cu_seq_lens     : list[int] = [0]
        kv_cu_seq_lens    : list[int] = [0]
        block_tables      : list[int] = []
        cu_blocks_lens    : list[int] = [0]
        new_cache_slots   : list[int] = []
        selected_token_ids: list[int] = []
        kv_seq_lens       : list[int] = []
        for sequence, q_seq_len in zip(batch_sequences, q_seq_lens):
            kv_seq_len = sequence.n_kv_cache_tokens + q_seq_len
            kv_seq_lens.append(kv_seq_len)
            for i in range(sequence.n_kv_cache_tokens, kv_seq_len):
                token_ids.append(sequence.token_ids[i])
                positions.append(i)
                new_cache_slots.append(sequence.block_table[i // self.block_size] * self.block_size + i % self.block_size)
            q_cu_seq_lens.append(q_cu_seq_lens[-1] + q_seq_len)
            kv_cu_seq_lens.append(kv_cu_seq_lens[-1] + kv_seq_len)
            block_tables += sequence.block_table
            cu_blocks_lens.append(cu_blocks_lens[-1] + len(sequence.block_table))
            if(kv_seq_len >= sequence.num_prompt_tokens):
                selected_token_ids.append(len(token_ids) - 1)
        input_ids    = torch.tensor(token_ids, dtype=torch.int, device=self.config.device)
        position_ids = torch.tensor(positions, dtype=torch.int, device=self.config.device)
        if len(pixel_values) > 0:
            pixel_values = torch.cat(pixel_values, dim=0).to(dtype=self.config.dtype, device=self.config.device)
        else:
            pixel_values = None
        
        t_q_cu_seq_lens = torch.tensor(q_cu_seq_lens, dtype=torch.int, device=self.config.device)
        t_kv_cu_seq_lens = torch.tensor(kv_cu_seq_lens, dtype=torch.int, device=self.config.device)
        t_new_cache_slots = torch.tensor(new_cache_slots, dtype=torch.int, device=self.config.device)
        t_block_tables = torch.tensor(block_tables, dtype=torch.int ,device=self.config.device)
        t_cu_blocks_lens = torch.tensor(cu_blocks_lens, dtype=torch.int, device=self.config.device)
        q_max_seq_len = max(q_seq_lens)
        kv_max_seq_len = max(kv_seq_lens)
        
        model_params = ModelParameters(
            attention_params=[AttentionParameters(
                kv_cache        = self.kv_caches[layer_id], 
                q_cu_seq_lens   = t_q_cu_seq_lens, 
                kv_cu_seq_lens  = t_kv_cu_seq_lens, 
                new_cache_slots = t_new_cache_slots, 
                block_tables    = t_block_tables, 
                cu_blocks_lens  = t_cu_blocks_lens, 
                q_max_seq_len   = q_max_seq_len, 
                kv_max_seq_len  = kv_max_seq_len, 
                num_sequences   = len(batch_sequences), 
                all_sequences_decode = False, 
            ) for layer_id in range(self.model_config.text_config.num_hidden_layers)]
        )

        # 3. forward
        logits = self.model_runner.forward(input_ids, pixel_values, position_ids, model_params)
        for sequence, q_seq_len in zip(batch_sequences, q_seq_lens):
            sequence.n_kv_cache_tokens += q_seq_len

        # 4. sample
        if len(selected_token_ids) > 0:
            sample_token_ids = torch.argmax(logits[selected_token_ids, :], dim=-1, keepdim=False).to(torch.device('cpu'))

            i = 0
            for sequence in batch_sequences:
                if sequence.is_decode:
                    token_id = sample_token_ids[i].item()
                    i += 1
                    sequence.token_ids.append(token_id)

        # 5. prepare next batch
        self.batch_policy.add_intermediate_sequences(batch_sequences)

        # 6. return
        for sequence in batch_sequences:
            if sequence.is_finished:
                finished_output.append(sequence)
            elif sequence.is_decode:
                decode_output.append(sequence)
        return decode_output, finished_output

    def _insert_image_token_ids(self, token_ids: list[int], image_token_id: int, num_token_insert: int):
        # we insert 575 image_token_id before each image_token_id
        inserted_token_ids = []
        for token_id in token_ids:
            if token_id == image_token_id:
                inserted_token_ids.extend([image_token_id] * num_token_insert)
            inserted_token_ids.append(token_id)
        
        return inserted_token_ids