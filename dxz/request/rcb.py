from dxz.engine.isa import Instruction, InstructionList
from dxz.memory.virtual_kv_cache import VirtualKVCache
from dxz.request.metric import RequestMetric
from dxz.request.request import SamplingParameters

class OutputTokenProcessor:
    def append_token_id(self, token_id: int, is_last_token: bool=False):
        raise Exception('interface not implemented')


class PrintOutputTokenProcessor(OutputTokenProcessor):
    def __init__(self):
        pass

    def append_token_id(self, token_id, is_last_token = False):
        print(f'output token {token_id}, {is_last_token}')


class LogOutputTokenProcessor(OutputTokenProcessor):
    def __init__(self):
        self.token_ids: list[int] = []

    def append_token_id(self, token_id, is_last_token = False):
        self.token_ids.append(token_id)

class RequestControlBlock:
    def __init__(self, instructions: InstructionList, n_virtual_kv_caches: int, sampling_params: SamplingParameters):
        self.instructions: InstructionList = instructions
        self.n_virtual_kv_caches = n_virtual_kv_caches
        self.virtual_kv_caches: list[VirtualKVCache] = []

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