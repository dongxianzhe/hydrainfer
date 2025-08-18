from typing import Literal, Any
from dataclasses import dataclass, field


class NodeType:
    def __init__(self, node_type: str = "EPD"):
        assert node_type in ["E", "P", "D", "EP", "ED", "PD", "EPD"], f"invalid node type {node_type}"
        self.node_type = node_type
        self.enable_encode = "E" in node_type
        self.enable_prefill = "P" in node_type
        self.enable_decode = "D" in node_type
        self.has_kv_cache = "P" in node_type or "D" in node_type
        self.has_image_cache = "E" in node_type or "P" in node_type
        self.has_vision_model = "E" in node_type
        self.has_language_model = "P" in node_type or "D" in node_type

    def __eq__(self, other: "NodeType"):
        if isinstance(other, NodeType):
            return self.node_type == other.node_type
        elif isinstance(other, str):
            return self.node_type == other
        return False


@dataclass
class MigrateNode:
    id: str # ray actor id
    tpot_slo: float # the TPOT SLO of the node expected
    actor: Any # the ray actor object


@dataclass
class MigrateGraph:
    ep_table: dict[str, list[MigrateNode]] = field(default_factory=dict)
    pd_table: dict[str, list[MigrateNode]] = field(default_factory=dict)

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
        # ray actor id -> MigrateNode
        self.enodes: dict[str, MigrateNode] = {}
        self.pnodes: dict[str, MigrateNode] = {}
        self.dnodes: dict[str, MigrateNode] = {}

    def add_node(self, actor: Any, tpot_slo: float, node_type: NodeType):
        for ty in node_type.node_type.lower():
            self._add_node(actor, tpot_slo, ty)

    def _add_node(self, actor: Any, tpot_slo: float, node_type: Literal['e', 'p', 'd']):
        nodes: dict[str, MigrateNode] = getattr(self, f"{node_type}nodes", None)
        if nodes is None:
            raise Exception(f'invalid node type {node_type}')

        actor_id = actor._actor_id.hex()
        migrate_node = MigrateNode(
            id=actor_id, 
            tpot_slo=tpot_slo,
            actor=actor,
        )
        nodes[actor_id] = migrate_node 

    def remove_node(self, actor: Any, node_type: NodeType):
        for ty in node_type.node_type.lower():
            self._remove_node(actor, ty)

    def remove_node(self, actor: Any, node_type: Literal['e', 'p', 'd']):
        nodes: dict[str, MigrateNode] = getattr(self, f"{node_type}nodes", None)
        if nodes is None: 
            raise Exception(f'invalid node type {node_type}')

        actor_id = actor._actor_id.hex()
        del nodes[actor_id]

    def build_graph(self) -> MigrateGraph:
        ep_table = {}
        pd_table = {}
        for id in self.enodes.keys():
            ep_table[id] = list(self.pnodes.values())
        for id in self.pnodes.keys():
            pd_table[id] = list(self.dnodes.values())
        return MigrateGraph(ep_table, pd_table)