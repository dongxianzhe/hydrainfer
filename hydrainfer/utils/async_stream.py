import asyncio
from typing import Union

class AsyncStream:
    def __init__(self, interval: float=0) -> None:
        self._queue: asyncio.Queue = asyncio.Queue()
        self._finished = False
        self.interval = interval

    def put(self, item: Union[str, Exception]) -> None:
        if self._finished:
            return
        self._queue.put_nowait(item)

    def finish(self) -> None:
        self._queue.put_nowait(StopAsyncIteration())
        self._finished = True

    @property
    def finished(self) -> bool:
        return self._finished

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        result = await self._queue.get()
        if isinstance(result, Exception):
            raise result
        if self.interval > 0:
            await asyncio.sleep(self.interval)
        return result
