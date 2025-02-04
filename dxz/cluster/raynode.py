from dxz.request.rcb import RequestControlBlock
from dxz.request.request import Request

class RayNode:
    async def add_request(self, request: Request):
        raise NotImplementedError

    async def step(self):
        raise NotImplementedError

    async def step_loop(self):
        raise NotImplementedError

    async def register_node(self, node: "RayNode"): 
        raise NotImplementedError

    async def migrate(self, rcb: RequestControlBlock):
        raise NotImplementedError