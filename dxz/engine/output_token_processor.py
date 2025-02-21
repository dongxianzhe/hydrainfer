import time
from tqdm import tqdm
from transformers import AutoTokenizer
from dataclasses import dataclass
from dxz.utils.async_stream import AsyncStream
from dxz.utils.counter import Counter
from dxz.request.offline_inference_output import OfflineInferenceOutput


class OutputTokenProcessor:
    def append_token_id(self, token_id: int, is_last_token: bool=False):
        raise NotImplementedError


class PrintOutputTokenProcessor(OutputTokenProcessor):
    def __init__(self):
        pass

    def append_token_id(self, token_id, is_last_token = False):
        print(f'output token {token_id}, {is_last_token}')


class PrintTextOutputTokenProcessor(OutputTokenProcessor):
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def append_token_id(self, token_id, is_last_token = False):
        print(f'output token {self.tokenizer.decode(token_id)}, {is_last_token}')


class LogOutputTokenProcessor(OutputTokenProcessor):
    def __init__(self):
        self.token_ids: list[int] = []

    def append_token_id(self, token_id, is_last_token = False):
        self.token_ids.append(token_id)


class OnlineStreamOutputTokenProcessor(OutputTokenProcessor):
    def __init__(self, tokenizer: AutoTokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.output_stream = AsyncStream()

    def append_token_id(self, token_id: int, is_last_token: bool=False):
        output_text = self.tokenizer.decode(token_id)
        self.output_stream.put(output_text)
        if is_last_token:
            self.output_stream.put(StopAsyncIteration())


class OnlineNonStreamOutputTokenProcessor(OutputTokenProcessor):
    def __init__(self, tokenizer: AutoTokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.output_tokens: list[int] = []
        self.output_stream = AsyncStream()
        
    def append_token_id(self, token_id: int, is_last_token: bool=False):
        self.output_tokens.append(token_id)
        if is_last_token:
            output_text = self.tokenizer.decode(self.output_tokens)
            self.output_stream.put(output_text)
            self.output_stream.put(StopAsyncIteration())


class OfflineOutputTokenProcessor(OutputTokenProcessor):
    def __init__(self, output: OfflineInferenceOutput, tokenizer: AutoTokenizer, counter: Counter, bar: tqdm):
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


class ZmqOutputTokenProcessor(OutputTokenProcessor):
    def __init__(self, request_id: int, stream: bool, offline: bool, zmq_send, tokenizer: AutoTokenizer):
        self.request_id = request_id
        self.stream = stream
        self.offline = offline
        self.zmq_send = zmq_send
        self.tokenizer = tokenizer

        self.arrival_time = time.perf_counter()
        self.output_token_ids: list[int] = []
        self.token_times: list[float] = []
        self.ttft: list[float] = []
        self.tpot: list[float] = []
        self.finished_time: float = 0

    def append_token_id(self, token_id: int, is_last_token: bool=False):
        self.output_token_ids.append(token_id)
        self.token_times.append(time.perf_counter())

        is_first_token: bool = len(self.output_token_ids) == 1
        if is_first_token:
            self.ttft = self.token_times[-1]
        else:
            self.tpot = self.token_times[-1] - self.token_times[-2]

        if is_last_token:
            self.finished_time = time.perf_counter()

        if self.offline:
            if self.stream:
                raise Exception('offline inference is not support stream output')
            else:
                if is_last_token:
                    self.zmq_send.send_pyobj(OfflineInferenceOutput(
                        text = self.tokenizer.decode(self.output_token_ids),
                        output_token_ids = self.output_token_ids, 
                        arrival_time  = self.arrival_time, 
                        finished_time = self.finished_time, 
                        token_times = self.token_times,
                        ttft = self.ttft, 
                        tpot = self.tpot, 
                    ))
        else:
            if self.stream:
                self.zmq_send.send_pyobj((self.request_id, [token_id]))
            else:
                if is_last_token:
                    self.zmq_send.send_pyobj((self.request_id, [token_id]))

@dataclass
class OutputTokenParams:
    print_output_text: bool = False
    is_stream_output: bool = False
    is_offline_output: bool = False