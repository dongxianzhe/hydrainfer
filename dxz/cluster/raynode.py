from dxz.request.rcb import RequestControlBlock

class RayNode:
    async def step(self):
        raise NotImplementedError

    async def step_loop(self):
        raise NotImplementedError

    async def register_node(self, node: "RayNode"): 
        raise NotImplementedError

    async def migrate(self, rcb: RequestControlBlock):
        raise NotImplementedError