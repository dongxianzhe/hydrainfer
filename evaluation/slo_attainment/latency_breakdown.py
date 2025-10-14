import jsonlines
from dataclasses import dataclass, field
from collections import defaultdict
from hydrainfer.cluster.log_server import LatencyBreakDownMetric
from dacite import from_dict

@dataclass
class RequestLatencyBreakdown:
    encode_queueing: float
    encode_execute: float
    ep_transfer: float
    prefill_queueing: float
    prefill_execute: float
    pd_transfer: float
    decode_queueing: float
    decode_execute: float
    metrics = list[LatencyBreakDownMetric]


class LatencyBreakdownAnalzer:
    def __init__(self):
        self.metrics: list[LatencyBreakDownMetric] = []

    def load_result(self, path: str):
        with open(path, 'r') as f:
            for line in jsonlines.Reader(f):
                metric: LatencyBreakDownMetric = from_dict(data_class=LatencyBreakDownMetric, data=line)
                self.metrics.append(metric)

    def parse(self):
        self.request_id_to_metrics: dict[str, list[LatencyBreakDownMetric]] = defaultdict(list)
        for metric in self.metrics:
            self.request_id_to_metrics[metric.request_id].append(metric)

        self.latnecybreakdowns: list[RequestLatencyBreakdown] = []
        for request_id, metrics in self.request_id_to_metrics.items():
            time_stamps: dict[str, list[float]] = defaultdict(list)
            for metric in metrics:
                time_stamps[metric.name].append(metric.time_stamp)
            is_success_request = all([name in time_stamps for name in ['arrival', 'IE', 'EPMR', 'PR', 'EF', 'PDMR', 'PR', 'TF']]) and len(time_stamps['PR']) == 2
            if is_success_request:
                self.latnecybreakdowns.append(RequestLatencyBreakdown(
                    encode_queueing=time_stamps['IE'][0] - time_stamps['arrival'][0], 
                    encode_execute=time_stamps['EPMR'][0] - time_stamps['IE'][0], 
                    ep_transfer=time_stamps['EF'][0] - time_stamps['PR'][0], 
                    prefill_queueing=time_stamps['PR'][0] - time_stamps['EPMR'][0], 
                    prefill_execute=time_stamps['PDMR'][0] - time_stamps['EF'][0], 
                    pd_transfer=time_stamps['TF'][0] - time_stamps['PR'][1], 
                    decode_queueing=time_stamps['PR'][1] - time_stamps['PDMR'][0], 
                    decode_execute=time_stamps['TF'][-1] - time_stamps['TF'][0], 
                ))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    argparse.ArgumentParser()
    parser.add_argument(f'--path', type=str, help='The path of jsonl log path.')
    args = parser.parse_args()
    analyzer = LatencyBreakdownAnalzer()
    analyzer.load_result(path=args.path)
    analyzer.parse()
    for latency_breakdown in analyzer.latnecybreakdowns:
        print(latency_breakdown)