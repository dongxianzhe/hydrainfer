import argparse
import torch
import time
import shortuuid
import json
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
from contextlib import asynccontextmanager
import asyncio
from typing import AsyncGenerator
from dxz.entrypoint.api_protocol import CompletionRequest, CompletionResponse, CompletionResponseChoice, CompletionResponseStreamChoice, CompletionStreamResponse, ChatCompletionRequest, ChatCompletionResponseStreamChoice, ChatCompletionStreamResponse, DeltaMessage
from dxz.engine.engine import EngineConfig
from dxz.engine.async_engine import AsyncEngine, AsyncEngineConfig

parser = argparse.ArgumentParser()
parser = AsyncEngineConfig.add_cli_args(parser)
args = parser.parse_args()
config = AsyncEngineConfig.from_cli_args(args)
async_engine = AsyncEngine(config)

@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    asyncio.create_task(async_engine.loop())
    yield
app = FastAPI(lifespan=lifespan)


@app.get('/health')
async def health() -> Response:
    return Response(status_code=200)

@app.post('/v1/chat/completions')
async def create_chat_completion(request: ChatCompletionRequest) -> Response:
    request_id = f"chatcmpl-{shortuuid.random()}"
    created_time = int(time.time())
    chunk_object_type = "chat.completion.chunk"

    result_generator = async_engine.message_generate(request.messages, request.max_tokens, request.stream)
    if request.stream:
        async def stream_results() -> AsyncGenerator:
            first_message_sent = set()
            async for output_text in result_generator:
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

@app.post('/v1/completions')
async def create_completion(request: CompletionRequest) -> Response:
    request_id = f"cmpl-{shortuuid.random()}"
    created_time = int(time.time())
    result_generator = async_engine.generate(request.prompt, request.stream)

    if request.stream:
        async def stream_results() -> AsyncGenerator:
            async for output_text in result_generator:
                response = CompletionStreamResponse(
                    id = request_id, 
                    object = "text_completion", 
                    created=created_time,
                    model=request.model,
                    choices = [
                        CompletionResponseStreamChoice(
                            index = 0, 
                            text = output_text
                        )]
                    )
                yield f"data: {response.model_dump_json(exclude_unset=True)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(stream_results())
    else:
        output_text = await result_generator.__anext__()
        choices = [CompletionResponseChoice(
            index=0, 
            text=output_text,
        )] # todo now only support one output
        return  CompletionResponse(
            id=request_id,
            object="text_completion",
            created=created_time,
            model=request.model,
            choices=choices,
        )

if __name__ == '__main__':
    uvicorn.run(
        app,
        host='127.0.0.1', 
        port=8888, 
        log_level='info'
    )