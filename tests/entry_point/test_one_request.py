import aiohttp
import asyncio
import json

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
        print('==========================header============================')
        print(headers)
        print('-------------------------- data ----------------------------')
        print(data)
        print('------------------------ response  -------------------------')
        async with session.post(url=url, headers=headers, json=data) as response:
            if response.status == 200:
                async for chunk, _ in response.content.iter_chunks():
                    print(chunk)
        print('============================================================')

async def main():
    url = 'http://127.0.0.1:8888/v1/completions'
    for i in range(1):
        await fetch(url)

asyncio.run(main())