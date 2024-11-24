from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
import uvicorn

import ray

from dxz.engine.llm_engine import AsyncLLMEngine
import asyncio

app = FastAPI()

next_engine_id: int = 0
async_llm_engines: list[AsyncLLMEngine] = [] # todo multi instances
init_instance = False

@app.get('/health')
async def health() -> Response:
    return Response(status_code=200)

@app.post('/v1/completions')
async def generate(request: Request) -> Response:
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")

    # global next_engine_id
    # print(f'{prompt} will be scheduled to {next_engine_id} instance')
    # instance scheduler
    # instance = async_llm_engines[next_engine_id]
    # next_engine_id = (next_engine_id + 1) % len(async_llm_engines)

    output = await async_llm_engines[0].add_request.remote(prompt)

    return JSONResponse({'text' : output})

if __name__ == '__main__':
    for i in range(1):
        instance = ray.remote(num_cpus=0, num_gpus=1)(AsyncLLMEngine).remote()
        async_llm_engines.append(instance)
        instance.start.remote()
        # instance = AsyncLLMEngine()
        # async_llm_engines.append(instance)
        
    uvicorn.run(
        app,
        host='127.0.0.1', 
        port=8888, 
        log_level='info'
    )