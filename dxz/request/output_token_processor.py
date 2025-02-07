from transformers import AutoTokenizer
from dxz.utils.async_stream import AsyncStream


class OutputTokenProcessor:
    def append_token_id(self, token_id: int, is_last_token: bool=False):
        raise NotImplementedError


class PrintOutputTokenProcessor(OutputTokenProcessor):
    def __init__(self):
        pass

    def append_token_id(self, token_id, is_last_token = False):
        print(f'output token {token_id}, {is_last_token}')


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