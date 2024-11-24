import asyncio
import queue
from dataclasses import dataclass, field
import torch
from torch import nn
from dxz.model.parameters import InputParameters
from dxz.memory.kv_cache import KVCache
from dxz.memory.block_allocator import BlockAllocator
from dxz.model_runner.cuda_graph_model_runner import CudaGraphModelRunner
from transformers import PreTrainedTokenizer
from dxz.model.downloader import download_hf_model
from dxz.entrypoint.async_stream import AsyncStream

@dataclass
class Sequence:
    id: int = 0
    token_ids: list[int] = field(default_factory=list)
    num_prompt_tokens: int = 0 # the number of prompt tokens
    n_kv_cache_tokens: int = 0 # the number of tokens already in kv cache

    block_table: list[int] = field(default_factory=list)
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

class ContinuousBatch:
    def __init__(self, max_tokens_per_batch: int, chunk_size: int, allocator: BlockAllocator, block_size: int):
        self.max_tokens_per_batch = max_tokens_per_batch
        self.chunk_size           = chunk_size
        self.new_queue    : queue.Queue = queue.Queue()
        self.prefill_queue: queue.Queue = queue.Queue()
        self.decode_queue : queue.Queue = queue.Queue()
        self.allocator = allocator
        self.block_size = block_size
    
    def add_new_sequence(self, sequence: Sequence):
        self.new_queue.put(sequence)

    def add_prefill_sequence(self, sequence: Sequence):
        self.prefill_queue.put(sequence)
    
    def add_decode_sequence(self, sequence: Sequence):
        self.decode_queue.put(sequence)

    def batch(self):
        done: bool = False
        batch_num_tokens = 0
        batch_sequences: list[Sequence] = []
        q_seq_lens = []
        while not self.decode_queue.empty():
            sequence: Sequence = self.decode_queue.get()
            
            if len(sequence.block_table) * self.block_size < len(sequence.token_ids): # need allocate new block
                blocks = self.allocator.allocate(1)
                if len(blocks) == 0: # out of memory
                    if not self.prefill_queue.empty(): # try to find an prefill sequence to preempt
                        preempted_sequence:Sequence = self.prefill_queue.get()
                        self.allocator.free(preempted_sequence.block_table)
                        preempted_sequence.n_kv_cache_tokens = 0
                        self.new_queue.put(preempted_sequence)
                        blocks = self.allocator.allocate(1)
                    elif not self.decode_queue.empty(): # try to find an decode sequence to preempt
                        preempted_sequence:Sequence = self.prefill_queue.get()
                        self.allocator.free(preempted_sequence.block_table)
                        preempted_sequence.n_kv_cache_tokens = 0
                        self.new_queue.put(preempted_sequence)
                        blocks = self.allocator.allocate(1)
                    else: # no sequence can be preempted, stop batch
                        self.decode_queue.put(sequence)
                        done = True
                        break
                sequence.block_table += blocks
            
            if batch_num_tokens + 1 <= self.max_tokens_per_batch:
                # select this sequence
                batch_num_tokens += 1 
                batch_sequences.append(sequence)
                q_seq_lens.append(1)

            else: # this batch has too many tokens
                done = True
                break
            
        while not done and (not self.prefill_queue.empty() or not self.new_queue.empty()):
            if not self.prefill_queue.empty():
                sequence: Sequence = self.prefill_queue.get()
            else:
                sequence: Sequence = self.new_queue.get()
            # compute chunk size
            num_chunk_tokens = min(self.chunk_size, len(sequence.token_ids) - sequence.n_kv_cache_tokens, self.max_tokens_per_batch - batch_num_tokens)
            if num_chunk_tokens == 0:
                if sequence.is_new:
                    self.new_queue.put(sequence)
                else:
                    self.prefill_queue.put(sequence)
                done = True
                break
            
            if len(sequence.block_table) * self.block_size <= sequence.n_kv_cache_tokens + num_chunk_tokens: # need allocate block
                n_blocks = (sequence.n_kv_cache_tokens + num_chunk_tokens - len(sequence.block_table) * self.block_size + self.block_size - 1) // self.block_size
                blocks = self.allocator.allocate(n_blocks)
                if len(blocks) == 0: # out of memory
                    self.prefill_queue.put(sequence)
                    done = True
                    break
                else:
                    sequence.block_table += blocks

            # select this sequence
            batch_num_tokens += num_chunk_tokens
            batch_sequences.append(sequence)
            q_seq_lens.append(num_chunk_tokens)
        
        return batch_sequences, q_seq_lens

def load_model_tokenizer(model_name: str, dtype: torch.dtype, device: torch.device) -> tuple[nn.Module, PreTrainedTokenizer]:
    if model_name == 'gpt2':
        from dxz.model.gpt2 import GPT2LMHeadModel
        from transformers import GPT2Tokenizer
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        n_kv_heads = model.config.n_head
        head_size = model.config.n_embd // model.config.n_head
        n_layers = model.config.n_layer
        max_seq_len  = model.config.n_positions
        model.to(dtype)
        model.to(device)
        model.eval()
        return model, tokenizer, n_kv_heads, head_size, n_layers, max_seq_len
    elif model_name == 'meta-llama/Llama-2-7b-hf':
        from dxz.model.llama import LlamaForCausalLM
        from transformers import LlamaTokenizer
        model_weights_path = download_hf_model(repo_id=model_name)
        model = LlamaForCausalLM.from_safetensor(model_weights_path, dtype=dtype, device=device)
        tokenizer = LlamaTokenizer.from_pretrained(model_weights_path)
        n_kv_heads = model.config.num_key_value_heads
        head_size = model.config.head_dim
        n_layers = model.config.num_hidden_layers
        max_seq_len  = model.config.max_position_embeddings
        return model, tokenizer, n_kv_heads, head_size, n_layers, max_seq_len
    elif model_name == 'fake':
        from dxz.model.fake import FakeModel
        from transformers import GPT2Tokenizer, GPT2Config
        config = GPT2Config.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = FakeModel(config)
        n_kv_heads = model.config.n_head
        head_size = model.config.n_embd // model.config.n_head
        n_layers = model.config.n_layer
        max_seq_len  = model.config.n_positions
        model.to(dtype)
        model.to(device)
        model.eval()
        return model, tokenizer, n_kv_heads, head_size, n_layers, max_seq_len
    else:
        raise Exception(f'invalid model {model_name}')

class LLMEngine:
    def __init__(self):
        self.device = torch.device('cuda:0')
        self.dtype = torch.half
        # 1. init model
        # self.model, self.tokenizer, self.n_kv_heads, self.head_size, self.n_layers, self.max_seq_len = load_model_tokenizer(model_name= 'gpt2', dtype=self.dtype, device=self.device)
        self.model, self.tokenizer, self.n_kv_heads, self.head_size, self.n_layers, self.max_seq_len = load_model_tokenizer(model_name='meta-llama/Llama-2-7b-hf', dtype=self.dtype, device=self.device)
        print(f'model info: n_kv_heads  {self.n_kv_heads} head_size   {self.head_size} n_layers    {self.n_layers} max_seq_len {self.max_seq_len}')
        print()
        # 2. init kv cache
        self.block_size = 16
        self.num_blocks = 4 * 1024 * 1024 * 1024 // 2 // self.n_layers // self.block_size // self.n_kv_heads // self.head_size 
        self.kv_caches = [KVCache(self.num_blocks, self.block_size, self.n_kv_heads, self.head_size, dtype=self.dtype, device=self.device) for _ in range(self.n_layers)]
        self.allocator = BlockAllocator(self.num_blocks)
        print(f'kv cache info block_size {self.block_size} num blocks {self.num_blocks}')
        # 3. capture cuda graph for fast decode
        self.model_runner = self.model
        # self.model_runner = CudaGraphModelRunner(model_runner=self.model, dtype=self.dtype,device=self.device,block_size=self.block_size, vocab_size=self.tokenizer.vocab_size, kv_caches=self.kv_caches, cuda_graph_max_batch_size=1, cuda_graph_max_seq_len=self.max_seq_len)
        
        # 4. batch policy
        self.sequence_id_allocator: int = 0
        self.batch_policy = ContinuousBatch(max_tokens_per_batch = 1024, chunk_size = 256, allocator=self.allocator, block_size=self.block_size)

    def add_request(self, prompt: str) -> int:
        # now it's user responsibility to not pass in prompt length greater than model can forward
        self.sequence_id_allocator += 1
        token_ids = self.tokenizer.encode(prompt)
        sequence = Sequence(
            id = self.sequence_id_allocator, 
            token_ids = token_ids, 
            num_prompt_tokens = len(token_ids), 
            eos_token_id = self.tokenizer.eos_token_id, 
            max_seq_len  = self.max_seq_len
        ) 
        self.batch_policy.add_new_sequence(sequence)

        return sequence.id

    @torch.inference_mode()
    def step(self) -> tuple[list[Sequence], list[Sequence]]:
        # 1. batch and allocate memory
        batch_sequences, q_seq_lens = self.batch_policy.batch()
        if len(batch_sequences) == 0:
            return [], []
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
            if kv_seq_len >= sequence.num_prompt_tokens:
                selected_token_ids.append(len(token_ids) - 1)

        input_ids = torch.tensor(token_ids, dtype=torch.int, device=self.device)
        position_ids = torch.tensor(positions, dtype=torch.int, device=self.device)
        input_params = InputParameters(
            num_sequences = len(batch_sequences), 
            q_cu_seq_lens = torch.tensor(q_cu_seq_lens, dtype=torch.int, device=self.device),
            kv_cu_seq_lens = torch.tensor(kv_cu_seq_lens, dtype=torch.int, device=self.device), 
            new_cache_slots = torch.tensor(new_cache_slots, dtype=torch.int, device=self.device), 
            block_tables = torch.tensor(block_tables, dtype=torch.int, device=self.device), 
            cu_blocks_lens = torch.tensor(cu_blocks_lens, dtype=torch.int, device=self.device), 
            q_max_seq_len = max(q_seq_lens), 
            kv_max_seq_len = max(kv_seq_lens)
        )
        
        # 3. forward
        logits = self.model_runner.forward(input_ids, position_ids, self.kv_caches, input_params)
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
            
        # 5. put sequnce batch to queue and free finished
        decode_output: list[Sequence] = []
        finished_output: list[Sequence] = []
        for sequence in batch_sequences:
            if sequence.is_prefill_but_not_new:
                self.batch_policy.add_prefill_sequence(sequence)
            elif sequence.is_decode_but_not_finished:
                decode_output.append(sequence)
                self.batch_policy.add_decode_sequence(sequence)
            elif sequence.is_finished:
                finished_output.append(sequence)
                self.allocator.free(sequence.block_table)
            else:
                raise Exception
        
        return decode_output, finished_output

class AsyncLLMEngine:
    def __init__(self) -> None:
        self.llm_engine = LLMEngine()
        self.is_stream_output:dict[int, bool]        = {} # sequence.id -> wheather stream output
        self.output_streams  :dict[int, AsyncStream] = {} # sequence.id -> output generator

    def generate(self, prompt: str, stream: bool) -> AsyncStream:
        id = self.llm_engine.add_request(prompt)

        output_stream = AsyncStream()
        self.is_stream_output[id] = stream
        self.output_streams  [id] = output_stream
        return output_stream

    async def loop(self):
        while True:
            decode, finished = self.llm_engine.step() 

            tokenizer = self.llm_engine.tokenizer
            for sequence in decode:
                if self.is_stream_output[sequence.id]:
                    output_text = tokenizer.decode(sequence.token_ids[-1])
                    output_stream = self.output_streams[sequence.id]
                    output_stream.put(output_text)
            for sequence in finished:
                if self.is_stream_output[sequence.id]:
                    output_text = tokenizer.decode(sequence.token_ids[-1])
                else:
                    output_text = tokenizer.decode(sequence.token_ids)
                output_stream = self.output_streams[sequence.id]
                output_stream.put(output_text)
                output_stream.put(StopAsyncIteration())
                del self.is_stream_output[sequence.id]
                del self.output_streams[sequence.id]

            await asyncio.sleep(0)