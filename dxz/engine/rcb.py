from typing import Optional
from dxz.engine import Instruction, InstructionList, OutputTokenProcessor, RequestMetric
from dxz.engine.output_token_processor import OutputTokenParams
from dxz.memory import VirtualTokenCache
from dxz.request import SamplingParameters


class RequestControlBlock:
    def __init__(self, request_id: int, instructions: InstructionList, sampling_params: SamplingParameters, output_token_params: OutputTokenParams, slo_stringent: bool=False):
        self.request_id = request_id
        self.instructions: InstructionList = instructions
        self.virtual_kv_cache: VirtualTokenCache = None
        self.virtual_image_cache: VirtualTokenCache = None

        self.sid: int = -1

        self.output_token_processors: list[OutputTokenProcessor] = []
        self.output_token_params = output_token_params
        self.output_token_ids: list[int] = []
        self.sampling_params = sampling_params
        self.metric = RequestMetric()

        self.slo_stringent = slo_stringent

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