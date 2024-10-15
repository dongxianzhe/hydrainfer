class Sequence:
    def __init__(self):
        self.id = 0
        self.num_prompt_tokens = 0
        self.token_ids: list[int] = []
        self.n_kv_cache_tokens = 0
        self.block_table: list[int] = []