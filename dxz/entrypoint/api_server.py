import time
import shortuuid
import json
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
from dxz.engine.async_llm_engine import AsyncLLMEngine
from contextlib import asynccontextmanager
import asyncio
from typing import AsyncGenerator
from dxz.entrypoint.api_protocol import CompletionRequest, CompletionResponse, CompletionResponseChoice

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
    print(request)
    request_id = f"cmpl-{shortuuid.random()}"
    created_time = int(time.time())
    result_generator = async_llm_engine.generate(request.prompt)

    # if stream:
    #     async def stream_results() -> AsyncGenerator:
    #         async for output_text in result_generator:
    #             ret = {"text" : prompt + output_text}
    #             print(ret)
    #             yield (json.dumps(ret) + "\0").encode('utf-8')
    #     return StreamingResponse(stream_results())
    # else:
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