import json
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
from dxz.engine.async_llm_engine import AsyncLLMEngine
from contextlib import asynccontextmanager
import asyncio
from typing import AsyncGenerator

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
async def generate(request: Request) -> Response:
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    print(f'stream = {stream}')
    result_generator = async_llm_engine.generate(prompt)

    if stream:
        async def stream_results() -> AsyncGenerator:
            async for output_text in result_generator:
                ret = {"text" : prompt + output_text}
                print('------------------------------------------------------------')
                print(ret)
                print('------------------------------------------------------------')
                yield (json.dumps(ret) + "\0").encode('utf-8')
        return StreamingResponse(stream_results())
    else:
        output = await result_generator.__anext__()
        return JSONResponse({'text' : output})

if __name__ == '__main__':
    uvicorn.run(
        app,
        host='127.0.0.1', 
        port=8888, 
        log_level='info'
    )