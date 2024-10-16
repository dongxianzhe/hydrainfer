from transformers import GPT2Tokenizer
from dxz.memory.block_allocator import BlockAllocator
from dxz.engine.llm_engine import LLMEngine
from dxz.request.sequence import Sequence
import asyncio

class ValidationError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__()
        self.code = code
        self.message = message

    def __repr__(self) -> str:
        return super().__repr__() + self.message

    def __str__(self) -> str:
        return super().__str__() + self.message

class OutputAsyncStream:
    """A stream of RequestOutput objects, which can be used to
    send responses to the client asynchronously."""

    def __init__(self) -> None:
        # asyncio.Queue is used to store the items in the stream, not thread-safe
        self._queue = asyncio.Queue()
        # event loop used to schedule callbacks from other threads
        self._loop = asyncio.get_running_loop()
        assert self._loop is not None
        self._cancelled = False

    def _put_nowait(self, item):
        # put item into asyncio.queue in a thread-safe way
        self._loop.call_soon_threadsafe(self._queue.put_nowait, item)

    # put item into the stream
    # None to indicate the end of the stream
    def put(self, item) -> bool:
        # if the stream is cancelled, return False
        if self._cancelled:
            return False

        # if item.status is not None and not item.status.ok:
        #     self._put_nowait(ValidationError(item.status.code, item.status.message))
        #     return False

        # put the item into the queue
        self._put_nowait(item)
        # if item.finished:
            # self._put_nowait(StopAsyncIteration())
        self._put_nowait(StopAsyncIteration())
        return True

    # report an error to the stream, rerais as an exception
    def error(self, error: str) -> bool:
        self._queue.put_nowait(Exception(error))
        return True

    # cancel the stream
    def cancel(self) -> None:
        self._cancelled = True
        self._queue.put_nowait(StopAsyncIteration())

    def __aiter__(self):
        return self

    # async generator to iterate over the stream
    async def __anext__(self):
        item = await self._queue.get()
        # reraise the exception
        if isinstance(item, Exception):
            raise item
        return item

class AsyncLLMEngine:
    def __init__(self) -> None:
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.llm_engine = LLMEngine()
        self.allocator = BlockAllocator(self.llm_engine.num_blocks) # refactor num blocks
        self.queue = asyncio.Queue(1000)

        self.sequence_id_allocator = 0

        self.callbacks = {}

    async def add_request(self, prompt: str) -> str:
        # 1. tokenize
        # 2. insert into queue
        # 3. register call back and return an awaitable thing to wait async engine call back
        sequence = Sequence() 
        sequence.id = self.sequence_id_allocator
        self.sequence_id_allocator += 1
        sequence.token_ids = self.tokenizer.encode(prompt)
        sequence.num_prompt_tokens = len(sequence.token_ids)
        sequence.block_table = self.allocator.allocate(5)

        await self.queue.put(sequence)

        output_stream = OutputAsyncStream()

        def callback(output: str) -> bool:
            return output_stream.put(output)

        self.callbacks[sequence.id] = callback
        output = await output_stream.__anext__()

        return output

    async def start(self):
        # 1. batch
        # 2. forward
        # 3. check output
        while True:
            batch: list[Sequence] = []
            while not self.queue.empty() and len(batch) < 100:
                s = await self.queue.get()
                batch.append(s)

            if len(batch) > 0:
                self.llm_engine.execute_model(batch)

                max_tokens = 50
                for sequence in batch:
                    def check_stop(sequence: Sequence) -> bool:
                        return len(sequence.token_ids) - sequence.num_prompt_tokens == max_tokens
                    if check_stop(sequence):
                        result = self.tokenizer.decode(sequence.token_ids)
                        self.callbacks[sequence.id](result)
                    else:
                        await self.queue.put(sequence)

            await asyncio.sleep(0)