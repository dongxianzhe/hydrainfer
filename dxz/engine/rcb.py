from typing import Optional
from dxz.engine import Instruction, InstructionList, OutputTokenProcessor, RequestMetric
from dxz.memory.virtual_kv_cache import VirtualKVCache
from dxz.request import SamplingParameters


class RequestControlBlock:
    def __init__(self, instructions: InstructionList, n_virtual_kv_caches: int, sampling_params: SamplingParameters):
        self.instructions: InstructionList = instructions
        self.n_virtual_kv_caches = n_virtual_kv_caches
        self.virtual_kv_caches: list[VirtualKVCache] = []
        self.virtual_token_cache: VirtualKVCache = None

        self.sid: int = -1

        self.output_token_processors: list[OutputTokenProcessor] = []
        self.sampling_params = sampling_params
        self.metric = RequestMetric()

    def current_instruction(self) -> Instruction:
        return self.instructions.curr

    def step(self):
        self.instructions.curr = self.instructions.curr.next

    def is_finished(self) -> bool:
        return self.instructions.curr is None

    def register_output_token_processor(self, output_token_processor: OutputTokenProcessor):
        self.output_token_processors.append(output_token_processor)

    def print(self):
        print(f'---------------------------- request control block --------------------------------')
        print(f'sid {self.sid}')
        for instruction in self.instructions:
            print(f"{instruction}")
        print(f'-----------------------------------------------------------------------------------')