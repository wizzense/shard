"""scheduler / control plane — light and replaceable.

fits the target model to the currently-joined (heterogeneous) gpus, orders them
into a pipeline preferring low-latency edges, tracks health, and reassigns blocks
when a node drops. holds no weights and no user data, so decentralizing it later
(rotating/elected) is a follow-up, not a rewrite. hosted by the c0mpute
orchestrator at first.
"""

from dataclasses import dataclass
from .node import LayerRange


@dataclass
class JoinedNode:
    node_id: str
    vram_gb: float
    rtt_ms: dict  # node_id -> measured rtt to other nodes


class Scheduler:
    def __init__(self, model: str, total_layers: int):
        self.model = model
        self.total_layers = total_layers
        self.nodes: dict[str, JoinedNode] = {}

    def register(self, node: JoinedNode) -> None:
        raise NotImplementedError  # phase 3

    def allocate(self) -> dict[str, LayerRange]:
        """assign each node a contiguous block that fits its vram, covering the stack.

        privacy note (docs/ARCHITECTURE.md#privacy): pin the embedding + final blocks
        to trusted/staked nodes, leave only deep middle blocks to untrusted volunteers.
        """
        raise NotImplementedError  # phase 0 (static) / phase 3 (dynamic+heterogeneous)

    def topology(self, coordinator_id: str, k: int | None = None) -> list[str]:
        """order nodes into the cheapest pipeline loop on the measured rtt mesh.

        the coordinator is the depot (entry hop out, direct-return hop back); the stage
        order is the min-latency Hamiltonian loop through it. with k set, also selects the
        best k of the joined nodes. see shard/topology.py for the solver.
        """
        from .topology import optimal_loop, select_and_order
        ids = [nid for nid in self.nodes if nid != coordinator_id]
        coord = self.nodes[coordinator_id]
        L = [[0.0 if a == b else self.nodes[a].rtt_ms[b] for b in ids] for a in ids]
        c_out = [coord.rtt_ms[a] for a in ids]
        c_in = [self.nodes[a].rtt_ms[coordinator_id] for a in ids]
        idx = list(range(len(ids)))
        order, _ = (select_and_order(idx, L, c_out, c_in, k) if k else
                    optimal_loop(idx, L, c_out, c_in))
        return [ids[i] for i in order]

    def on_drop(self, node_id: str) -> None:
        """node died mid-generation: reassign its block, rebuild pipeline, retry in-flight."""
        raise NotImplementedError  # phase 3/4
