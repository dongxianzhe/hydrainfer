from dxz.engine.isa import Instruction, Fill
from dxz.memory.virtual_kv_cache import VirtualKVCache

class Sequence:
    def __init__(self, sid: int, instructions: list[Instruction], virtual_kv_caches: list[VirtualKVCache], max_tokens: int, eos_token_id: int, max_seq_len: int):
        self.sid: int = sid
        self.pc: int = 0
        self.instructions: list[Instruction] = instructions
        self.n_virtual_kv_caches: int
        self.virtual_kv_caches: list[VirtualKVCache] = virtual_kv_caches

        self.output_token_ids: list[int] = []
        self.max_tokens = max_tokens
        self.eos_token_id = eos_token_id
        self.max_seq_len = max_seq_len
        self.append_instruction_count = 0

    def next_instruction(self) -> Instruction:
        inst = self.instructions[self.pc]
        self.pc += 1
        return inst

    def append_instruction(self, instruction: Fill): # todo move to a proper place
        assert isinstance(instruction, Fill) and len(instruction.token_ids) == 1
        if instruction.token_ids[-1] != self.eos_token_id and len(self.output_token_ids) < self.max_tokens:
            self.instructions.append(instruction)
            self.append_instruction_count += 1

    def is_finished(self) -> bool:
        return self.pc >= len(self.instructions)

    def print(self):
        print(f'---------------------------- sequence info --------------------------------')
        print(f'sid {self.sid}')
        for instruction in self.instructions:
            print(f"{instruction}")
        print('----------------------------------------------------------------------------')