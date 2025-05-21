import time
import shortuuid
from dataclasses import dataclass, field
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
from contextlib import asynccontextmanager
import asyncio
from typing import AsyncGenerator, Optional
from dxz.entrypoint.api_protocol import CompletionRequest, CompletionResponse, CompletionResponseChoice, CompletionResponseStreamChoice, CompletionStreamResponse, ChatCompletionRequest, ChatCompletionResponseStreamChoice, ChatCompletionStreamResponse, DeltaMessage, ChatCompletionMessage
from dxz.utils.async_stream import AsyncStream
from dxz.request import Request, SamplingParameters
from dxz.engine import RequestProcessParameters, OutputTokenParams
from dxz.model import ModelFactoryConfig, getModelFactory, ModelFactoryContext
from dxz.utils.zmq_utils import ZMQConfig, init_zmq_recv


class RequestObserver:
    def add_request(self, request: Request, params: RequestProcessParameters):
        raise NotImplementedError


@dataclass
class APIServerConfig:
    host: str = "127.0.0.1"
    port: int = 8888
    zmq: ZMQConfig = field(default_factory=ZMQConfig)
    model: ModelFactoryConfig = field(default_factory=ModelFactoryConfig)

class APIServer:
    def __init__(self, config: APIServerConfig):
        self.config = config
        model_factory = getModelFactory(config.model, ModelFactoryContext())
        self.processor = model_factory.getProcessor()
        self.tokenizer = model_factory.getTokenizer()
        self.vision_config = model_factory.getVisionModelConfig()

        self.zmq_recv = init_zmq_recv(config.zmq)
        @asynccontextmanager
        async def lifespan(fastapi_app: FastAPI):
            asyncio.create_task(self._zmq_recv_loop())
            yield
        self.app = FastAPI(lifespan=lifespan)
        self._register_routes()
        self.async_streams: dict[str, AsyncStream] = {}
        self.observers :list[RequestObserver] = []

    async def _zmq_recv_loop(self):
        while True:
            request_id, output_text = await self.zmq_recv.recv_pyobj()
            # print(f'zmq recv {request_id} {output_text}')
            if request_id in self.async_streams:
                output_stream = self.async_streams[request_id]
                output_stream.put(output_text)
                if output_text is None:
                    del self.async_streams[request_id]
                    output_stream.finish()
                    print(f'request {request_id} finished')
            await asyncio.sleep(0)

    def register(self, observer: RequestObserver):
        self.observers.append(observer)

    def _parse_content(self, messages: list[ChatCompletionMessage]) -> list[str]: 
        # parch each content image_url with <image> token
        # return a list of base64 image
        image_base64_list: list[str] = []
        for message in messages:
            text_content: str = ""
            image_content: str = ""
            for content in message.content:
                if content.type == 'text':
                    text_content += content.text
                elif content.type == 'image_url':
                    image_content += self.vision_config.image_token
                    prefix, image_base64 = content.image_url.url.split(',')
                    assert prefix == 'data:image/png;base64', f'only support base 64 png image url but got {prefix}'
                    image_base64_list.append(image_base64)
            message.content = image_content + '\n' + text_content
        return image_base64_list

    def _register_routes(self):
        @self.app.get('/health')
        async def health() -> Response:
            return Response(status_code=200)

        @self.app.post('/v1/chat/completions')
        async def create_chat_completion(request: ChatCompletionRequest) -> Response:
            request_id = f"chatcmpl-{shortuuid.random()}"
            created_time = int(time.time())
            chunk_object_type = "chat.completion.chunk"

            assert len(request.messages) == 1, 'only support one round chat'

            image_base64_list = self._parse_content(request.messages)
            assert len(image_base64_list) == 1, f'only support one image per request, but got {len(image_base64_list)} images'
            prompt = self.tokenizer.apply_chat_template(request.messages)
            
            async_stream = AsyncStream()
            self.async_streams[request_id] = async_stream
            for observer in self.observers:
                observer.add_request(Request(
                    request_id = request_id, 
                    prompt = prompt, 
                    image = None, 
                    image_base64 = image_base64_list[0], 
                    sampling_params = SamplingParameters(max_tokens=request.max_tokens)
                ), RequestProcessParameters(
                    outout_token_parmas = OutputTokenParams(
                        print_output_text = False, 
                        is_stream_output = True, 
                        is_offline_output = False, 
                    )
                ))

            if request.stream:
                async def stream_results() -> AsyncGenerator:
                    first_message_sent = set()
                    async for output_text in async_stream:
                        index = 0
                        if index not in first_message_sent:
                            response = ChatCompletionStreamResponse(
                                id = request_id, 
                                object = chunk_object_type, 
                                created=created_time,
                                model=request.model,
                                choices = [
                                    ChatCompletionResponseStreamChoice(
                                        index = index, 
                                        delta = DeltaMessage(
                                            role = "assistant", 
                                            content = ""
                                        )
                                    )]
                                )
                            first_message_sent.add(index)
                            yield f"data: {response.model_dump_json(exclude_unset=True)}\n\n"
                        if output_text:
                            response = ChatCompletionStreamResponse(
                                id = request_id, 
                                object = chunk_object_type, 
                                created=created_time,
                                model=request.model,
                                choices = [ChatCompletionResponseStreamChoice(
                                    index = index, 
                                    delta = DeltaMessage(content = output_text)
                                )]
                            )
                            yield f"data: {response.model_dump_json(exclude_unset=True)}\n\n"

                    yield "data: [DONE]\n\n"
                return StreamingResponse(stream_results())
            else:
                raise Exception('not support non stream chat completion')

    def run(self):
        print('api server is running')
        uvicorn.run(self.app, host=self.config.host, port=self.config.port, log_level='info')

# @app.post('/v1/chat/completions')
# async def create_chat_completion(request: ChatCompletionRequest) -> Response:
#     request_id = f"chatcmpl-{shortuuid.random()}"
#     created_time = int(time.time())
#     chunk_object_type = "chat.completion.chunk"

#     result_generator = async_engine.message_generate(request.messages, request.max_tokens, request.stream)
#     if request.stream:
#         async def stream_results() -> AsyncGenerator:
#             first_message_sent = set()
#             async for output_text in result_generator:
#                 index = 0
#                 if index not in first_message_sent:
#                     response = ChatCompletionStreamResponse(
#                         id = request_id, 
#                         object = chunk_object_type, 
#                         created=created_time,
#                         model=request.model,
#                         choices = [
#                             ChatCompletionResponseStreamChoice(
#                                 index = index, 
#                                 delta = DeltaMessage(
#                                     role = "assistant", 
#                                     content = ""
#                                 )
#                             )]
#                         )
#                     first_message_sent.add(index)
#                     yield f"data: {response.model_dump_json(exclude_unset=True)}\n\n"
#                 if output_text:
#                     response = ChatCompletionStreamResponse(
#                         id = request_id, 
#                         object = chunk_object_type, 
#                         created=created_time,
#                         model=request.model,
#                         choices = [ChatCompletionResponseStreamChoice(
#                             index = index, 
#                             delta = DeltaMessage(content = output_text)
#                         )]
#                     )
#                     yield f"data: {response.model_dump_json(exclude_unset=True)}\n\n"
#             yield "data: [DONE]\n\n"
#         return StreamingResponse(stream_results())
#     else:
#         raise Exception('not support non stream chat completion')