from PIL import Image
from typing import Optional
from dxz.engine.engine import Engine
from dxz.entrypoint.async_stream import AsyncStream
from dxz.entrypoint.api_protocol import ChatCompletionMessage
from dxz.engine.request import Request
import base64
import asyncio
import io
class AsyncEngine:
    def __init__(self, config: Engine):
        self.engine = Engine(config=config)
        self.tokenizer = self.engine.tokenizer
        self.is_stream_output:dict[int, bool]        = {} # sequence.id -> wheather stream output
        self.output_streams  :dict[int, AsyncStream] = {} # sequence.id -> output generator
        self.request_id = 0

    def message_generate(self, messages: list[ChatCompletionMessage], max_tokens: int, stream: bool):
        assert len(messages) == 1, 'only support single round conversation'
        id = self.request_id
        self.request_id += 1
        self.engine.add_request(Request(
            prompt = messages[0].content, 
            image = None, 
            image_base64 = messages[0].image, 
            max_tokens = max_tokens, 
            request_id = id, 
            ))

        output_stream = AsyncStream()
        self.is_stream_output[id] = stream
        self.output_streams  [id] = output_stream
        return output_stream
    
    async def loop(self):
        while True:
            output_tokens = self.engine.step()
            # if len(output_tokens) > 0:
            #     print(f'output_tokenss: {output_tokens}')
            for rid, token_id in output_tokens.items():
                if self.is_stream_output[rid]:
                    output_text = self.tokenizer.decode(token_id)
                    output_stream = self.output_streams[rid]
                    output_stream.put(output_text)

            finished = self.engine.scheduler.pop_finished()
            for seq in finished:
                if not self.is_stream_output[rid]:
                    output_text = self.tokenizer.decode(seq.output_token_ids)
                    output_stream = self.output_streams[seq.rid]
                    output_text.put(output_text)
                output_stream = self.output_streams[seq.rid]
                output_stream.put(StopAsyncIteration())
                del self.is_stream_output[seq.rid]
                del self.output_streams[seq.rid]
            await asyncio.sleep(0.001)