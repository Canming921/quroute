"""LearnedRouter — Stage B: a trained policy (GNN+RL) as a drop-in BaseRouter.

Now implemented: it wraps any trained policy (e.g. `quroute.train.gnn.GNNMaskedPolicy`)
and routes by running the RoutingEnv under that policy. The GNN policy is size-agnostic,
so a single trained model routes on topologies it was never trained on.

    from quroute.train.gnn import train_gnn, GNNMaskedPolicy
    from quroute import LearnedRouter, grid_topology
    net = train_gnn([grid_topology(3, 3)], iterations=400)
    router = LearnedRouter(GNNMaskedPolicy(net))    # use inside QurouteRouter

Design (state/action/reward) is documented in docs/physics and train/gnn.py.

References
----------
* Li, Ding, Xie, "Tackling the Qubit Mapping Problem ..." (SABRE), ASPLOS 2019
* Pozzi et al., "Using Reinforcement Learning to Perform Qubit Routing ..."
"""
from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from .base import BaseRouter, RoutedResult
from .policy_router import PolicyRouter


class LearnedRouter(BaseRouter):
    name = "learned_rl_gnn"

    def __init__(self, policy=None, *, max_steps: int | None = None):
        if policy is None:
            raise NotImplementedError(
                "LearnedRouter needs a trained policy, e.g. "
                "GNNMaskedPolicy(train_gnn(...)). Use GreedyShortestPathRouter for a "
                "no-training baseline."
            )
        self._router = PolicyRouter(policy, max_steps=max_steps)

    def route(
        self,
        circuit: QuantumCircuit,
        coupling_map: CouplingMap,
        initial_layout: dict[int, int] | None = None,
    ) -> RoutedResult:
        return self._router.route(circuit, coupling_map, initial_layout)
