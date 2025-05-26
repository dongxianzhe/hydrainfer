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

    def _table_repr(self, table: dict[str, list[MigrateNode]]) -> str:
        lines = []
        for i, nodes in table.items():
            line = f'{i}'
            for j in nodes:
                line += f' -> {j.id}'
            lines.append(line)
        return '\n'.join(lines)

    def __repr__(self) -> str:
        return (
            '------------------- migrate graph ---------------------\n'
            'ep table:\n'
            f'{self._table_repr(self.ep_table)}\n'
            'pd table:\n'
            f'{self._table_repr(self.pd_table)}\n'
            '------------------------------------------------------'
        )

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