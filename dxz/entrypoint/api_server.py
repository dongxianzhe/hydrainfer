from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
import uvicorn

from dxz.engine.llm_engine import LLMEngine

app = FastAPI()
llm_engine = LLMEngine()

@app.get('/health')
async def health() -> Response:
    return Response(status_code=200)

@app.post('/generate')
async def generate(request: Request) -> Response:
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    results = llm_engine.generate(prompts=[prompt])
    return JSONResponse({'text' : results[0]})

if __name__ == '__main__':
    uvicorn.run(
        app,
        host='127.0.0.1', 
        port=8888, 
        log_level='info'
    )