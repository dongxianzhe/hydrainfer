from PIL import Image
from typing import Optional
from dxz.engine.engine import EngineConfig
from dxz.cluster.epdnode import EPDNode
from dxz.entrypoint.async_stream import AsyncStream
from dxz.entrypoint.api_protocol import ChatCompletionMessage
from dxz.request.request import Request, SamplingParameters
from dxz.request.rcb import OutputTokenProcessor
from transformers import AutoTokenizer
import base64
import asyncio
import io

class OnlineStreamOutputTokenProcessor(OutputTokenProcessor):
    def __init__(self, tokenizer: AutoTokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.output_stream = AsyncStream()

    def append_token_id(self, token_id: int, is_last_token: bool=False):
        output_text = self.tokenizer.decode(token_id)
        self.output_stream.put(output_text)
        if is_last_token:
            self.output_stream.put(StopAsyncIteration())


class OnlineNonStreamOutputTokenProcessor(OutputTokenProcessor):
    def __init__(self, tokenizer: AutoTokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.output_tokens: list[int] = []
        self.output_stream = AsyncStream()
        
    def append_token_id(self, token_id: int, is_last_token: bool=False):
        self.output_tokens.append(token_id)
        if is_last_token:
            output_text = self.tokenizer.decode(self.output_tokens)
            self.output_stream.put(output_text)
            self.output_stream.put(StopAsyncIteration())


class AsyncEngine:
    def __init__(self, config: EngineConfig):
        self.node = EPDNode(config=config)
        self.tokenizer = self.node.tokenizer
        self.request_id_allocator = 0

    def message_generate(self, messages: list[ChatCompletionMessage], max_tokens: int, stream: bool):
        assert len(messages) == 1, 'only support single round conversation'
        request_id = self.request_id_allocator
        self.request_id_allocator += 1

        if stream:
            processor = OnlineStreamOutputTokenProcessor(tokenizer=self.tokenizer)
        else:
            processor = OnlineNonStreamOutputTokenProcessor(tokenizer=self.tokenizer)

        self.node.add_request(Request(
            prompt = messages[0].content, 
            image = None, 
            image_base64 = messages[0].image, 
            request_id = request_id, 
            sampling_params=SamplingParameters(
                max_tokens=max_tokens
            )),
            processor
        )

        return processor.output_stream
    
    async def loop(self):
        while True:
            self.node.step()
            await asyncio.sleep(0.001)