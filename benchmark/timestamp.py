import csv
import json
import functools
from datetime import datetime
from typing import Literal, Iterable, Callable
import numpy as np


class PoissonProcessIntervals:
    def __init__(self, request_rate: float):
        self.request_rate = request_rate 
    
    def __getitem__(self, index: int):
        if self.request_rate == float('inf'):
            return 0
        return np.random.exponential(1.0 / self.request_rate)


def get_mooncake_timestamps(file_path: str) -> list[float]:
    timestamps: list[float] = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            timestamps.append(data['timestamp'])
    return timestamps


def get_azure_timestamps(file_path: str) -> list[float]:
    timestamps: list[float] = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        first_timestamp = None
        for row in reader:
            timestamp_str = row['TIMESTAMP'][:26]
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
            if first_timestamp is None:
                first_timestamp = timestamp
                timestamps.append(0.0)
            else:
                delta = timestamp - first_timestamp
                timestamps.append(delta.total_seconds())
    return timestamps


def get_burstgpt_timestamp(file_path: str) -> list[float]:
    timestamps = []
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            timestamps.append(float(row['Timestamp']))
    return timestamps


class RealLoadIntervals:
    def __init__(self, get_timestamps: Callable[[], list[float]], request_rate: float):
        self.request_rate = request_rate
        timestamps = get_timestamps()
        self.n_requests = len(timestamps)
        timestamp_scale = (timestamps[-1] - timestamps[0]) / (self.n_requests / self.request_rate)
        self.timestamps = [(timestamp - timestamps[0]) / timestamp_scale for timestamp in timestamps]
        print(self.n_requests)

    def __getitem__(self, index: int):
        index %= self.n_requests
        if index == 0:
            return 0
        return self.timestamps[index] - self.timestamps[index - 1]


def get_intervals(method: Literal['poisson'], request_rate: float) -> Iterable:
    if method == 'poisson':
        return PoissonProcessIntervals(request_rate)
    elif method == 'azure_code': 
        return RealLoadIntervals(get_timestamps=functools.partial(get_azure_timestamps, file_path="/datasets/AzureLLMInferenceTrace_code.csv"), request_rate=request_rate)
    elif method == 'azure_conv':
        return RealLoadIntervals(get_timestamps=functools.partial(get_azure_timestamps, file_path="/datasets/AzureLLMInferenceTrace_conv.csv"), request_rate=request_rate)
    elif method == 'burstgpt':
        return RealLoadIntervals(get_timestamps=functools.partial(get_burstgpt_timestamp, file_path="/datasets/BurstGPT_without_fails.csv"), request_rate=request_rate)
    elif method == 'mooncake':
        return RealLoadIntervals(get_timestamps=functools.partial(get_mooncake_timestamps, file_path="/datasets/mooncake_conv_trace.jsonl"), request_rate=request_rate)
    else:
        raise Exception(f'invalid method {method}')