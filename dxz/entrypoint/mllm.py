from dataclasses import dataclass
from dxz.engine.request import Request
from dxz.engine.engine import EngineConfig
from dxz.request.rcb import RequestControlBlock
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
        finished: list[RequestControlBlock] = []
        bar = tqdm(range(len(requests)))
        while len(finished) < len(requests):
            self.node.step()
            f = self.node.engine.scheduler.pop_finished() 
            finished += f
            bar.update(len(f))

        finished_time = time.perf_counter()
        finished = sorted(finished, key=lambda seq: seq.sid)

        for rcb in finished:
            outputs.append(GenerateOutput(
                input_len = rcb.static_info.n_prompt_tokens, 
                text = self.tokenizer.decode(rcb.output_token_ids, skip_special_tokens=True), 
                arrival_time = rcb.metric.arrival_time, 
                finished_time = finished_time, 
                token_times = rcb.metric.tokens_time, 
            ))

        return outputs