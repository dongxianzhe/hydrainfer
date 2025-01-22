from dxz.cluster.scheduler import RequestScheduler


class NodeRegistry:
    def __init__(self):
        self.nodes = []
        self.schedulers = []

    def register_node(self, node):
        self.nodes.append(node)
        for scheduler in self.schedulers:
            scheduler.update_nodes(self.nodes)

    def register_scheduler(self, scheduler: RequestScheduler):
        self.schedulers.append(scheduler)
        scheduler.update_nodes(self.nodes)