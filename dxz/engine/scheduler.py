from queue import Queue
from typing import Literal, Optional
from dxz.sequence.sequence import Sequence
from dataclasses import dataclass, fields
from dxz.engine.isa import Instruction, Fill, TextFill, ImageFill, Mov, ReAlloc, EmptyInstruction, ImageEmbedFill, ImageEmbed
import time
import argparse

@dataclass
class SchedulerConfig:
    batch_policy: Literal['nobatch', 'requestlevel', 'continuousbatch'] = 'continuousbatch'
    priority: Literal['prefill', 'decode'] = 'prefill'
    max_running_sequences: int = 15
    max_batch_fill_tokens: int = 1024
    max_batch_embed_images: int = 3
    batch_embed_fill: bool = False
    debug_mode: bool = False

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> 'SchedulerConfig':
        attrs = [attr.name for attr in fields(cls)]
        config = cls(**{attr: getattr(args, attr) for attr in attrs})
        return config

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--batch_policy', type=str, choices=['nobatch', 'requestlevel', 'continuousbatch'], default='continuousbatch', help='Batch policy for scheduling.')
        parser.add_argument('--priority', type=str, choices=['prefill', 'decode'], default='prefill', help='Prefill prioritize or decode prioritize')
        parser.add_argument('--max-running-sequences', type=int, default=15, help='Maximum number of sequences running concurrently. other sequences will waiting in queue.')
        parser.add_argument('--max-batch-fill-tokens', type=int, default=1024, help='Maximum number of tokens in each batch fill.')
        parser.add_argument('--max-batch-embed-images', type=int, default=3, help='Maximum number of images to embed in each batch.')
        parser.add_argument('--batch-embed-fill', action='store_true', help='Enable batch embedding fill.')
        parser.add_argument('--debug_mode', action='store_true', help='Enable debug mode for more detailed logging.')
        return parser


class SequenceScheduler:
    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.waiting = Queue()
        self.running: list[Sequence] = []
        self.finished: list[Sequence] = []
        self.step_cnt = 0
    
    def schedule_new(self, sequences: list[Sequence]):
        for sequence in sequences:
            self.waiting.put(sequence)
    
    def schedule_running(self, sequences: list[Sequence]):
        self.running += sequences

    def schedule_unfinished(self, sequences: list[Sequence]):
        self.running += sequences
    
    def schedule_finished(self, sequences: list[Sequence]):
        self.finished += sequences

    def pop_finished(self) -> list[Sequence]:
        finished = self.finished
        self.finished = []
        return finished

    def step(self) -> list[tuple[Sequence, Instruction]]:
        self.step_cnt += 1
        schedule_time = time.perf_counter()
        # 1. get enough sequences to participate in the batch
        if self.config.batch_policy == 'nobatch':
            if len(self.running) == 0:
                if not self.waiting.empty():
                    sequence = self.waiting.get()
                    self.running.append(sequence)
        elif self.config.batch_policy == 'requestlevel':
            if len(self.running) == 0:
                while len(self.running) < self.config.max_running_sequences and not self.waiting.empty():
                    sequence = self.waiting.get()
                    self.running.append(sequence)
        elif self.config.batch_policy == 'continuousbatch':
            while len(self.running) < self.config.max_running_sequences and not self.waiting.empty():
                sequence = self.waiting.get()
                self.running.append(sequence)
        if len(self.running) == 0:
            return []

        batch_fill_tokens = 0
        batch_embed_images = 0
        prefill_seqs: list[Sequence] = []
        decode_seqs : list[Sequence] = []
        embed_seqs  : list[Sequence] = []
        next_step: list[Sequence] = []
        this_step: list[Sequence] = []
        for sequence in self.running:
            inst = sequence.curr_instruction()
            if isinstance(inst, Fill):
                if len(inst.token_ids) == 1:
                    decode_seqs.append(sequence)
                else:
                    prefill_seqs.append(sequence)
            elif isinstance(inst, ImageEmbed):
                embed_seqs.append(sequence)
            else:
                this_step.append(sequence)


        if len(prefill_seqs) > 0 and not self.config.batch_embed_fill:
            next_step += embed_seqs
        else:
            for seq in embed_seqs:
                if batch_embed_images < self.config.max_batch_embed_images:
                    this_step.append(seq)
                    batch_embed_images += 1 # todo cope with multi image
                else:
                    next_step.append(seq)

        fill_seqs = prefill_seqs + decode_seqs if self.config.priority == 'prefill' else decode_seqs + prefill_seqs
            
        for seq in fill_seqs:
            inst = seq.curr_instruction()
            if batch_fill_tokens < self.config.max_batch_fill_tokens:
                this_step.append(seq)
                batch_fill_tokens += len(inst.token_ids)
            else:
                next_step.append(seq)

        if self.config.debug_mode:
            print(f'------------------------------ scheduler step {self.step_cnt} ------------------------------')
            print(f'sid : ' + ' '.join(f'{seq.sid: 2}'                 for seq in this_step))
            print(f'pc  : ' + ' '.join(f'{seq.pc : 2}'                 for seq in this_step))
            print(f'inst: ' + ' '.join(f'{seq.curr_instruction()}' for seq in this_step))

        for seq in this_step:
            if seq.metric.first_schedule_time == 0.:
                seq.metric.first_schedule_time = schedule_time

        self.running = next_step
        return [(seq, seq.next_instruction()) for seq in this_step]

    def __repr__(self):
        return f'{len(self.waiting)} {len(self.running)} {len(self.finished)}'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = SchedulerConfig.add_cli_args(parser)
    args = parser.parse_args()
    config = SchedulerConfig.from_cli_args(args)
    print(config)