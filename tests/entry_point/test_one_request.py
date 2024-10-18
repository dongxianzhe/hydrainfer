import aiohttp
import asyncio
import json

prompt = "hello"

async def fetch(url):
    async with aiohttp.ClientSession() as session:
        headers = {
            'Content-Type': 'application/json'
        }
        data = {
            "model": "gpt2",
            "prompt":"Hello",
            "max_tokens": 50,
            "temperature": 0, 
            "stream": True,
        }
        async with session.post(url=url, headers=headers, json=data) as response:
            print(type(response))
            if response.status == 200:
                async for chunk, _ in response.content.iter_chunks():
                    print(chunk)

async def main():
    url = 'http://127.0.0.1:8888/v1/completions'
    for i in range(1):
        await fetch(url)

asyncio.run(main())