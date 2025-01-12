from dataclasses import dataclass
from dxz.engine.request import Request
from dxz.engine.engine import EngineConfig
from dxz.sequence.sequence import Sequence
from dxz.cluster.epdnode import EPDNode
from tqdm import tqdm
import time

@dataclass
class GenerateOutput:
    input_len: int
    text : str
    arrival_time: float
    finished_time: float
    token_times: list[float]

class MLLM:
    def __init__(self, config: EngineConfig):
        self.node = EPDNode(config)
        self.tokenizer = self.node.tokenizer

    def generate(self, requests: list[Request]) -> list[GenerateOutput]:
        for request in requests:
            self.node.add_request(request)

        outputs = []
        finished: list[Sequence] = []
        bar = tqdm(range(len(requests)))
        while len(finished) < len(requests):
            self.node.step()
            f = self.node.engine.scheduler.pop_finished() 
            finished += f
            bar.update(len(f))

        finished_time = time.perf_counter()
        finished = sorted(finished, key=lambda seq: seq.sid)

        for sequence in finished:
            outputs.append(GenerateOutput(
                input_len = sequence.static_info.n_prompt_tokens, 
                text = self.tokenizer.decode(sequence.output_token_ids, skip_special_tokens=True), 
                arrival_time = sequence.metric.arrival_time, 
                finished_time = finished_time, 
                token_times = sequence.metric.tokens_time, 
            ))

        return outputs