from dataclasses import dataclass, field

@dataclass
class Sequence:
    id: int = 0
    token_ids: list[int] = field(default_factory=list)
    num_prompt_tokens: int = 0 # the number of prompt tokens
    n_kv_cache_tokens: int = 0 # the number of tokens already in kv cache

    block_table: list[int] = field(default_factory=list)
    # stop criterian
    max_tokens: int = 50
    eos_token_id:int = 0
    max_seq_len:int = 1024

    @property
    def is_finished(self) -> bool:
        return self.token_ids[-1] == self.eos_token_id or (len(self.token_ids) - self.num_prompt_tokens) == self.max_tokens or len(self.token_ids) > self.max_seq_len

    @property
    def is_prefill(self) -> bool:
        return self.n_kv_cache_tokens < self.num_prompt_tokens
    
    @property
    def is_prefill_but_not_new(self) -> bool:
        return self.n_kv_cache_tokens < self.num_prompt_tokens and not self.is_new

    @property
    def is_decode(self) -> bool:
        return self.n_kv_cache_tokens >= self.num_prompt_tokens
    
    @property
    def is_decode_but_not_finished(self) -> bool:
        return self.n_kv_cache_tokens >= self.num_prompt_tokens and not self.is_finished

    @property
    def is_new(self) -> bool:
        return self.n_kv_cache_tokens == 0