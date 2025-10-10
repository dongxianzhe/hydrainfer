import time
from typing import Optional
from hydrainfer.engine import Instruction, InstructionList, OutputTokenProcessor, ScenarioType
from hydrainfer.engine.output_token_processor import OutputTokenParams
from hydrainfer.memory import VirtualTokenCache
from hydrainfer.request import SamplingParameters, Request, RequestMetaData


class RequestControlBlock:
    def __init__(self):
        self.request_id: Optional[int] = None
        self.sampling_params: Optional[SamplingParameters] = None

        self.request_metadata: Optional[RequestMetaData] = None
        self.instructions: Optional[InstructionList] = None
        self.virtual_kv_cache: Optional[VirtualTokenCache] = None
        self.virtual_image_cache: Optional[VirtualTokenCache] = None
        self.output_token_processors: list[OutputTokenProcessor] = []
        self.output_token_params: Optional[OutputTokenParams] = None
        self.output_token_ids: list[int] = []
        self.scenario_type: Optional[ScenarioType] = None

        self.arrival_time = time.perf_counter()
        self.event_timestamps: list[tuple[str, float]] = []

    def current_instruction(self) -> Instruction:
        return self.instructions.curr

    def step(self):
        self.instructions.curr = self.instructions.curr.next

    def is_finished(self) -> bool:
        if self.instructions.curr is None:
            return True

        if len(self.output_token_ids) == self.sampling_params.max_tokens:
            return True

        if len(self.output_token_ids) > 0:
            for eos_token_id in self.sampling_params.eos_token_ids:
                if self.output_token_ids[-1] == eos_token_id:
                    return True

        return False

    def register_output_token_processor(self, output_token_processor: OutputTokenProcessor):
        self.output_token_processors.append(output_token_processor)
    
    def __repr__(self) -> str:
        lines = [
            '---------------------------- request control block --------------------------------',
            f'sid {self.sid}'
        ]
        lines.extend(repr(instruction) for instruction in self.instructions)
        lines.append('-----------------------------------------------------------------------------------')
        return '\n'.join(lines)


class BatchRequest:
    def __init__(self, rcbs: Optional[list[RequestControlBlock]] = None):
        self.rcbs = rcbs if rcbs is not None else []

    def __len__(self):
        return len(self.rcbs)
    
    def __getitem__(self, idx: int) -> tuple[RequestControlBlock, Instruction]:
        return self.rcbs[idx], self.rcbs[idx].instructions.curr

    def append(self, rcb: RequestControlBlock):
        self.rcbs.append(rcb)

    def step(self):
        for rcb in self.rcbs:
            rcb.step()