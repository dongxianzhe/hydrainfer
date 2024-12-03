from dxz.engine.isa import Instruction, TextFill
from dxz.memory.virtual_kv_cache import VirtualKVCache
from dxz.memory.compiler import CompilerOutput
from dataclasses import dataclass, field


@dataclass
class SequenceMetric:
    arrival_time: float = 0. 
    tokens_time: list[float] = field(default_factory=list)
    finished_time: float = 0.

class Sequence:
    def __init__(self, static_info: CompilerOutput, sid: int, instructions: list[Instruction], virtual_kv_caches: list[VirtualKVCache], max_tokens: int, eos_token_id: int, max_seq_len: int):
        self.static_info = static_info
        self.sid: int = sid
        self.pc: int = 0
        self.instructions: list[Instruction] = instructions
        self.virtual_kv_caches: list[VirtualKVCache] = virtual_kv_caches

        self.output_token_ids: list[int] = []
        self.max_tokens = max_tokens
        self.eos_token_id = eos_token_id
        self.max_seq_len = max_seq_len
        self.append_instruction_count = 0
        self.metric = SequenceMetric()

    def next_instruction(self) -> Instruction:
        inst = self.instructions[self.pc]
        self.pc += 1
        return inst

    def is_finished(self) -> bool:
        finished: bool = self.pc >= len(self.instructions)
        if finished:
            for kv_cache in self.virtual_kv_caches:
                kv_cache.realloc(0)
        return finished

    def print(self):
        print(f'---------------------------- sequence info --------------------------------')
        print(f'sid {self.sid}')
        for instruction in self.instructions:
            print(f"{instruction}")
        print('----------------------------------------------------------------------------')