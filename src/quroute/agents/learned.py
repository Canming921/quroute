"""LearnedRouter —— Stage B:把训练好的策略(GNN+RL)包装成即插即用的 BaseRouter。

现已实现:它包装任意训练好的策略(例如 `quroute.train.gnn.GNNMaskedPolicy`),
通过在 RoutingEnv 上运行该策略来完成路由。GNN 策略是尺寸无关的,因此一个训练好
的模型可以在它从未训练过的拓扑上路由。

    from quroute.train.gnn import train_gnn, GNNMaskedPolicy
    from quroute import LearnedRouter, grid_topology
    net = train_gnn([grid_topology(3, 3)], iterations=400)
    router = LearnedRouter(GNNMaskedPolicy(net))    # 可放进 QurouteRouter 使用

状态 / 动作 / 奖励的设计见 docs/physics 与 train/gnn.py。

参考文献
--------
* Li, Ding, Xie, "Tackling the Qubit Mapping Problem ..."(SABRE),ASPLOS 2019
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
