from dataclasses import dataclass, field


@dataclass
class VirtualKVCache:
    vid: int
    n_kv_cache_tokens: int = 0
    block_table: list[int] = field(default_factory=list)