from transformers import GPT2Tokenizer
from dxz.memory.block_allocator import BlockAllocator
from dxz.engine.llm_engine import LLMEngine
from dxz.request.sequence import Sequence
from dxz.entrypoint.async_stream import AsyncStream
import queue
import asyncio
class AsyncLLMEngine:
    def __init__(self) -> None:
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.llm_engine = LLMEngine()
        self.allocator = BlockAllocator(self.llm_engine.num_blocks) # refactor num blocks
        self.queue = queue.Queue()

        self.sequence_id_allocator = 0

        self.output_streams = {}

    def generate(self, prompt: str) -> AsyncStream:
        # 1. tokenize
        # 2. insert into queue
        # 3. register call back and return an awaitable thing to wait async engine call back
        sequence = Sequence() 
        sequence.id = self.sequence_id_allocator
        self.sequence_id_allocator += 1
        sequence.token_ids = self.tokenizer.encode(prompt)
        sequence.num_prompt_tokens = len(sequence.token_ids)
        sequence.block_table = self.allocator.allocate(5)

        self.queue.put(sequence)

        output_stream = AsyncStream()
        self.output_streams[sequence.id] = output_stream
        return output_stream

    async def loop(self):
        # 1. batch
        # 2. forward
        # 3. check output
        while True:
            batch: list[Sequence] = []
            while not self.queue.empty() and len(batch) < 100:
                s = self.queue.get()
                batch.append(s)

            if len(batch) > 0:
                self.llm_engine.execute_model(batch)

                max_tokens = 50
                for sequence in batch:
                    def check_stop(sequence: Sequence) -> bool:
                        return len(sequence.token_ids) - sequence.num_prompt_tokens == max_tokens
                    if check_stop(sequence):
                        result = self.tokenizer.decode(sequence.token_ids)
                        output_stream = self.output_streams[sequence.id]
                        output_stream.put(result)
                        output_stream.put(StopAsyncIteration())
                        del self.output_streams[sequence.id]
                    else:
                        self.queue.put(sequence)

            await asyncio.sleep(0)