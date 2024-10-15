from transformers import GPT2Tokenizer
from dxz.memory.block_allocator import BlockAllocator
from dxz.engine.llm_engine import LLMEngine
from dxz.request.sequence import Sequence
import asyncio

class AsyncLLMEngine:
    def __init__(self) -> None:
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.llm_engine = LLMEngine()
        self.allocator = BlockAllocator(self.llm_engine.num_blocks) # refactor num blocks
        self.queue = asyncio.Queue()

    async def add_request(self, prompt: str):
        # 1. tokenize
        # 2. insert into queue
        # 3. return call back?
        sequence = Sequence() 
        sequence.token_ids = self.tokenizer.encode(prompt)
        sequence.num_prompt_tokens = len(sequence.token_ids)
        sequence.block_table = self.allocator.allocate(5)
        await self.queue.put(sequence)

        # output = asyncio.Queue()
        # return output

    async def start(self):
        # 1. batch
        # 2. forward
        # 3. check output
        while True:
            batch: list[Sequence] = []
            while not self.queue.empty() and len(batch) < 10:
                s = await self.queue.get()
                batch.append(s)

            if len(batch) > 0:
                self.llm_engine.execute_model(batch)

                max_tokens = 50
                for sequence in batch:
                    def check_stop(sequence: Sequence) -> bool:
                        return len(sequence.token_ids) - sequence.num_prompt_tokens == max_tokens
                    if check_stop(sequence):
                        print(self.tokenizer.decode(sequence.token_ids))
                    else:
                        await self.queue.put(sequence)

            await asyncio.sleep(0)