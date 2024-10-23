from dataclasses import dataclass, field

@dataclass
class Sequence:
    id: int = 0
    num_prompt_tokens: int = 0
    token_ids: list[int] = field(default_factory=list)

    n_kv_cache_tokens: int = 0 # the number of tokens already in kv cache
    block_table: list[int] = field(default_factory=list) # the sequence's block table