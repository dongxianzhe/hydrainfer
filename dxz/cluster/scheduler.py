import random
from dataclasses import dataclass
from dxz.request.request import Request
from typing import Literal


@dataclass
class RequestScheudlerConfig:
    load_balance_policy: Literal['round', 'random'] = 'round'


class RequestScheduler:
    def __init__(self, config: RequestScheudlerConfig):
        self.config = config
        self.nodes = []
        self.next_node_id: int = 0

    def add_request(self, request: Request):
        node = self.nodes[self.next_node_id]
        print(f'schedule to node {self.next_node_id}')
        node.add_request.remote(request)
        if self.config.load_balance_policy == 'round':
            self.next_node_id = (self.next_node_id + 1) % len(self.nodes)
        elif self.config.load_balance_policy == 'random':
            self.next_node_id = random.randint(0, len(self.nodes) - 1)
        else:
            raise Exception('invalid load balance policy')

    def update_nodes(self, nodes): 
        self.nodes = nodes