class Sequence:
    def __init__(self):
        self.token_ids: list[int] = []
        self.n_kv_cache_tokens = 0
        self.block_table: list[int] = []