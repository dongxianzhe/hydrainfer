from dataclasses import dataclass, field

@dataclass
class RequestMetric:
    arrival_time: float = 0. 
    token_times: list[float] = field(default_factory=list)
    finished_time: float = 0.

    # used in latency breakdown, after inference, each list has two float record when start and end time respectively
    encode_queueing: list[float] = field(default_factory=list)
    encode_execute: list[float] = field(default_factory=list)
    ep_transfer: list[float] = field(default_factory=list)
    prefill_queueing: list[float] = field(default_factory=list)
    prefill_execute: list[float] = field(default_factory=list)
    pd_transfer: list[float] = field(default_factory=list)
    decode_queueing: list[float] = field(default_factory=list)
    decode_execute: list[float] = field(default_factory=list)

def log_latency_breakdown(metric: RequestMetric):
    if len(metric.encode_queueing ) > 0: 
        print(f"encode_queueing  time: {metric.encode_queueing [1] - metric.encode_queueing [0]}")
    if len(metric.encode_execute  ) > 0: 
        print(f"encode_execute   time: {metric.encode_execute  [1] - metric.encode_execute  [0]}")
    if len(metric.ep_transfer     ) > 0: 
        print(f"ep_transfer      time: {metric.ep_transfer     [1] - metric.ep_transfer     [0]}")
    if len(metric.prefill_queueing) > 0: 
        print(f"prefill_queueing time: {metric.prefill_queueing[1] - metric.prefill_queueing[0]}")
    if len(metric.prefill_execute ) > 0: 
        print(f"prefill_execute  time: {metric.prefill_execute [1] - metric.prefill_execute [0]}")
    if len(metric.pd_transfer     ) > 0: 
        print(f"pd_transfer      time: {metric.pd_transfer     [1] - metric.pd_transfer     [0]}")
    if len(metric.decode_queueing ) > 0: 
        print(f"decode_queueing  time: {metric.decode_queueing [1] - metric.decode_queueing [0]}")
    if len(metric.decode_execute  ) > 0: 
        print(f"decode_execute   time: {metric.decode_execute  [1] - metric.decode_execute  [0]}")