import random
from typing import Literal, Optional
from dataclasses import dataclass

class Worker:
    pass


@dataclass
class LoadBalancerConfig:
    policy: Literal['round', 'random'] = 'round'


class LoadBalancer:
    def __init__(self, config: LoadBalancerConfig):
        self.config = config
        self.workers: list[Worker] = []

        if self.config.policy == 'round':
            self.next_worker = 0

    def register_worker(self, worker: Worker):
        self.workers.append(worker)

    def choice(self) -> Optional[Worker]:
        # assert len(self.workers) > 0, f'number of workers {len(self.workers)} should be greater than 0'
        if len(self.workers) == 0:
            return None

        if self.config.policy == 'round':
            worker = self.workers[self.next_worker]
            self.next_worker = (self.next_worker + 1) % len(self.workers)
        elif self.config.policy == 'random':
            worker = self.workers[random.randint(0, len(self.workers) - 1)]
        else:
            raise Exception(f'invalid policy {self.config.policy}')
        return worker

    def __repr__(self) -> str:
        return f"loadbalancer of {self.workers}"

class CompositeLoadBlancer:
    def __init__(self):
        self.loadbalancers: dict[any, LoadBalancer] = {}

    def register_loadbalancer(self, key: any, loadbalancer: LoadBalancer):
        assert key not in self.loadbalancers
        self.loadbalancers[key] = loadbalancer

    def register_worker(self, key: any, worker: Worker):
        assert key in self.loadbalancers
        self.loadbalancers[key].register_worker(worker)
        
    def choice(self, key: any) -> Optional[Worker]:
        worker = self.loadbalancers[key].choice()
        if worker is not None:
            return worker
        
        for key, loadbalancer in self.loadbalancers.items():
            worker = loadbalancer.choice()
            if worker is not None:
                return worker
        return None

    def __repr__(self) -> str:
        return f'{self.loadbalancers}'