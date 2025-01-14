from transformers import AutoTokenizer
from dataclasses import dataclass, field
from dxz.engine.engine import EngineConfig
from dxz.request.rcb import RequestControlBlock, OutputTokenProcessor
from dxz.request.request import Request
from dxz.cluster.epdnode import EPDNode
from tqdm import tqdm
import time

class Counter:
    def __init__(self):
        self.cnt = 0
    
    def count(self):
        self.cnt += 1

    def value(self) -> int:
        return self.cnt

@dataclass
class GenerateOutput:
    text : str = ""
    output_token_ids: list[int] = field(default_factory=list)
    arrival_time: float = -1
    finished_time: float = -1
    token_times: list[float] = field(default_factory=list)
    ttft: float = -1
    tpot: list[float] = field(default_factory=list)
class OfflineOutputTokenProcessor(OutputTokenProcessor):
    def __init__(self, output: GenerateOutput, tokenizer: AutoTokenizer, counter: Counter, bar: tqdm):
        super().__init__()
        self.output = output
        self.tokenizer = tokenizer
        self.counter = counter
        self.bar = bar

    def append_token_id(self, token_id: int, is_last_token: bool=False):
        self.output.output_token_ids.append(token_id)
        self.output.token_times.append(time.perf_counter())

        is_first_token: bool = len(self.output.output_token_ids) == 1
        if is_first_token:
            self.output.ttft = self.output.token_times[-1]
        else:
            self.output.tpot = self.output.token_times[-1] - self.output.token_times[-2]

        if is_last_token:
            self.output.finished_time = time.perf_counter()
            self.output.text = self.tokenizer.decode(self.output.output_token_ids, skip_special_tokens=True)
            self.counter.count()
            self.bar.update(1)

class MLLM:
    def __init__(self, config: EngineConfig):
        self.node = EPDNode(config)
        self.tokenizer = self.node.tokenizer

    def generate(self, requests: list[Request]) -> list[GenerateOutput]:
        counter = Counter()
        bar = tqdm(range(len(requests)))
        arrival_time = time.perf_counter()
        outputs: list[GenerateOutput] = [GenerateOutput(arrival_time=arrival_time) for _ in range(len(requests))]
        for output, request in zip(outputs, requests):
            processor = OfflineOutputTokenProcessor(output, self.tokenizer, counter, bar)
            self.node.add_request(request, processor)
        
        while counter.value() < len(requests):
            self.node.step()

        return outputs