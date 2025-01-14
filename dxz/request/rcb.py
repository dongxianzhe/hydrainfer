from dxz.engine.isa import Instruction, TextFill
from dxz.memory.virtual_kv_cache import VirtualKVCache
from dxz.request.metric import RequestMetric
from dxz.request.request import SamplingParameters

class OutputTokenProcessor:
    def append_token_id(self, token_id: int, is_last_token: bool=False):
        raise Exception('interface not implemented')


class RequestControlBlock:
    def __init__(self, instructions: list[Instruction], n_virtual_kv_caches: int, sampling_params: SamplingParameters, output_token_processor: OutputTokenProcessor):
        self.pc: int = 0
        self.instructions: list[Instruction] = instructions
        self.n_virtual_kv_caches = n_virtual_kv_caches
        self.virtual_kv_caches: list[VirtualKVCache] = []

        self.sid: int = -1

        self.output_token_processor = output_token_processor
        self.sampling_params = sampling_params
        self.metric = RequestMetric()

    def is_finished(self) -> bool:
        finished: bool = self.pc >= len(self.instructions)

        if finished:
            for kv_cache in self.virtual_kv_caches:
                kv_cache.realloc(0)
        return finished

    def print(self):
        print(f'---------------------------- request control block --------------------------------')
        print(f'sid {self.sid}')
        for instruction in self.instructions:
            print(f"{instruction}")
        print(f'-----------------------------------------------------------------------------------')