from transformers import GPT2Tokenizer
from dxz.engine.llm_engine import LLMEngine
from dxz.request.sequence import Sequence
from dxz.entrypoint.async_stream import AsyncStream
import queue
import asyncio
class AsyncLLMEngine:
    def __init__(self) -> None:
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.llm_engine = LLMEngine()
        self.queue = queue.Queue()

        self.sequence_id_allocator = 0

        self.output_streams = {}

    def generate(self, prompt: str) -> AsyncStream:
        self.sequence_id_allocator += 1
        token_ids = self.tokenizer.encode(prompt)
        sequence = Sequence(
            id = self.sequence_id_allocator, 
            token_ids = token_ids, 
            num_prompt_tokens = len(token_ids)
        ) 
        self.queue.put(sequence)

        output_stream = AsyncStream()
        self.output_streams[sequence.id] = output_stream
        return output_stream

    async def loop(self):
        while True:
            # 1. batch
            batch: list[Sequence] = []
            while not self.queue.empty() and len(batch) < 100:
                s = self.queue.get()
                batch.append(s)

            # 2. forward
            if len(batch) > 0:
                finished_sequences, unfinished_sequences = self.llm_engine.execute_model(batch)

            # 3. response
                for sequence in finished_sequences:
                    result = self.tokenizer.decode(sequence.token_ids)
                    output_stream = self.output_streams[sequence.id]
                    output_stream.put(result)
                    output_stream.put(StopAsyncIteration())
                    del self.output_streams[sequence.id]
                for sequence in unfinished_sequences:
                    self.queue.put(sequence)
            await asyncio.sleep(0)