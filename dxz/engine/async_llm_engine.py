from dxz.engine.llm_engine import LLMEngine
from dxz.entrypoint.async_stream import AsyncStream
import asyncio
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