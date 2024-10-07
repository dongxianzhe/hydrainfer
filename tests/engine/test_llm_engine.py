# from dxz.engine.llm_engine import LLMEngine
from dxz.engine.async_llm_engine import AsyncLLMEngine
import asyncio

if __name__ == '__main__':
    llm_engine = AsyncLLMEngine()
    results = asyncio.run(llm_engine.generate(["Hello"]))
    for result in results:
        print(result)