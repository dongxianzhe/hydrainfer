from dataclasses import dataclass
from dxz.utils.config_util import CLIConfig


class Worker:
    pass


@dataclass
class LoadBalancerConfig(CLIConfig):
    pass


class LoadBalancer:
    def __init__(self, config: LoadBalancerConfig, workers: list[Worker]):
        self.config = config
        assert len(workers) > 0, f'number of workers {len(workers)} should be greater than 0'
        self.workers = workers
        self.num_workers = len(self.workers)
        self.next_worker = 0

    def register_worker(self, worker: Worker):
        self.workers.append(worker)
        self.num_workers += 1

    def choice(self) -> Worker:
        worker = self.workers[self.next_worker]
        self.next_worker = (self.next_worker + 1) % self.num_workers
        return worker