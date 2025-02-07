from dxz.request import Request
from dxz.engine import RequestControlBlock


class Engine:
    def add_request(request: Request):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError


class AsyncEngine:
    async def add_request(self, request: Request):
        raise NotImplementedError

    async def step(self):
        raise NotImplementedError

    async def step_loop(self):
        raise NotImplementedError

    async def register_node(self, node: "AsyncEngine"): 
        raise NotImplementedError

    async def migrate(self, rcb: RequestControlBlock):
        raise NotImplementedError