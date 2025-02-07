import time
from tqdm import tqdm
from transformers import AutoTokenizer
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