from dataclasses import dataclass, field


@dataclass
class VirtualKVCache:
    vid: int
    n_kv_cache_tokens: int = 0
    block_table: list[int] = field(default_factory=list)
    layer_id: int = -1
    memory_handle: list[int] = None