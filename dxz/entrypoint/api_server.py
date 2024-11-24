import time
import shortuuid
import json
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
from dxz.engine.llm_engine import AsyncLLMEngine
from contextlib import asynccontextmanager
import asyncio
from typing import AsyncGenerator
from dxz.entrypoint.api_protocol import CompletionRequest, CompletionResponse, CompletionResponseChoice, CompletionResponseStreamChoice, CompletionStreamResponse


async_llm_engine = AsyncLLMEngine()
@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    asyncio.create_task(async_llm_engine.loop())
    yield
app = FastAPI(lifespan=lifespan)


@app.get('/health')
async def health() -> Response:
    return Response(status_code=200)

@app.post('/v1/completions')
async def generate(request: CompletionRequest) -> Response:
    request_id = f"cmpl-{shortuuid.random()}"
    created_time = int(time.time())
    result_generator = async_llm_engine.generate(request.prompt, request.stream)

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