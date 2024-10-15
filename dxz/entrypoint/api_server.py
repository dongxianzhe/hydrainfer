from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
import uvicorn

import ray

from dxz.engine.async_llm_engine import AsyncLLMEngine

app = FastAPI()

next_engine_id: int = 0
async_llm_engines: list[AsyncLLMEngine] = [] # todo multi instances

@app.get('/health')
async def health() -> Response:
    return Response(status_code=200)

@app.post('/v1/completions')
async def generate(request: Request) -> Response:
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    global next_engine_id
    print(f'{prompt} will be scheduled to {next_engine_id} instance')

    # instance scheduler
    instance = async_llm_engines[next_engine_id]
    next_engine_id = (next_engine_id + 1) % len(async_llm_engines)

    await instance.add_request.remote(prompt)
    return JSONResponse({'text' : "todo return result"})

if __name__ == '__main__':
    for i in range(1):
        instance = ray.remote(num_cpus=0, num_gpus=1)(AsyncLLMEngine).remote()
        async_llm_engines.append(instance)
        instance.start.remote()
        
    uvicorn.run(
        app,
        host='127.0.0.1', 
        port=8888, 
        log_level='info'
    )