from typing import Literal, Any
from dataclasses import dataclass


@dataclass
class MigrateNode:
    id: str # ray actor id
    tpot_slo: float # the TPOT SLO of the node expected
    actor: Any # the ray actor object


@dataclass
class MigrateGraph:
    ep_table: dict[str, list[MigrateNode]]
    pd_table: dict[str, list[MigrateNode]]

    def print_table(self, table: dict[str, list[MigrateNode]]):
        for i, table in table.items():
            print(f'{i}', end='')
            for j in table:
                print(f' -> {j.id}', end='')
            print()

    def print(self):
        print('------------------- migrate graph ---------------------')
        print('ep table:')
        self.print_table(self.ep_table)
        print('pd table:')
        self.print_table(self.pd_table)


class MigrateGraphBuilder:
    def __init__(self):
        self.enodes: list[MigrateNode] = []
        self.pnodes: list[MigrateNode] = []
        self.dnodes: list[MigrateNode] = []

    def add_node(self, actor: Any, tpot_slo: float, node_type: Literal['e', 'p', 'd']):
        nodes = getattr(self, f"{node_type}nodes", None)
        if nodes is None:
            raise Exception(f'invalid node type {node_type}')

        nodes.append(MigrateNode(
            id=actor._actor_id.hex(),
            tpot_slo=tpot_slo,
            actor=actor,
        ))

    def build_graph(self) -> MigrateGraph:
        ep_table = {}
        pd_table = {}
        for e in self.enodes:
            ep_table[e.id] = self.pnodes
        for p in self.pnodes:
            pd_table[p.id] = self.dnodes
        return MigrateGraph(ep_table, pd_table)