from dataclasses import dataclass, field, fields

@dataclass
class OfflineInferenceOutput:
    text : str = ""
    output_token_ids: list[int] = field(default_factory=list)
    arrival_time: float = -1
    finished_time: float = -1
    token_times: list[float] = field(default_factory=list)
    ttft: float = -1
    tpot: list[float] = field(default_factory=list)