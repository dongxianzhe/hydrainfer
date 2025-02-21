from dataclasses import dataclass, field

@dataclass
class RequestMetric:
    arrival_time: float = 0. 
    first_schedule_time: float = 0.
    token_times: list[float] = field(default_factory=list)
    finished_time: float = 0.