from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
import uvicorn

import ray

from dxz.engine.async_llm_engine import AsyncLLMEngine

app = FastAPI()

next_engine_id: int = 0
async_llm_engines: list[AsyncLLMEngine] = []

@app.get('/health')
async def health() -> Response:
    return Response(status_code=200)

@app.post('/generate')
async def generate(request: Request) -> Response:
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    global next_engine_id
    print(f'{prompt} will be scheduled to {next_engine_id} instance')

    instance = async_llm_engines[next_engine_id]
    next_engine_id = (next_engine_id + 1) % 2
    ref = instance.generate.remote(prompts=[prompt])
    results = ray.get(ref)
    return JSONResponse({'text' : results[0]})

if __name__ == '__main__':
    for i in range(2):
        instance = ray.remote(num_cpus=0, num_gpus=1)(AsyncLLMEngine).remote()
        async_llm_engines.append(instance)
        
    uvicorn.run(
        app,
        host='127.0.0.1', 
        port=8888, 
        log_level='info'
    )