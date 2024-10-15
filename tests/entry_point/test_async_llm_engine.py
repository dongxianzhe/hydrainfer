from dxz.engine.async_llm_engine import AsyncLLMEngine
import ray
import asyncio

# instance = ray.remote(num_cpus=0, num_gpus=1)(AsyncLLMEngine).remote()

async def main():
    instance = AsyncLLMEngine()
    obj = asyncio.create_task(instance.start())
    output = await instance.add_request("Hello")
    print(output)
    await obj

if __name__ == '__main__':
    asyncio.run(main())