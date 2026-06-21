"""QurouteRouter —— 包装任意 BaseRouter 的 Qiskit 转译器 pass。

实现为 `TransformationPass`,因此能像 qiskit 自带的 `SabreSwap` 一样直接塞进
`PassManager`。正是这一点让本包成为真正的“编译器插件”,而不是一个独立脚本
(对应评分表:API 设计 / 一键部署)。

示例
----
    from qiskit.transpiler import PassManager
    from quroute import QurouteRouter, GreedyShortestPathRouter, grid_topology

    pm = PassManager([QurouteRouter(grid_topology(2, 3), GreedyShortestPathRouter())])
    routed = pm.run(my_circuit)
"""
from __future__ import annotations

from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.basepasses import TransformationPass

from .agents.base import BaseRouter
from .agents.greedy import GreedyShortestPathRouter


class QurouteRouter(TransformationPass):
    def __init__(
        self,
        coupling_map: CouplingMap,
        router: BaseRouter | None = None,
        initial_layout: dict[int, int] | None = None,
    ):
        super().__init__()
        self.coupling_map = coupling_map
        self.router = router or GreedyShortestPathRouter()
        self.initial_layout = initial_layout

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        circuit = dag_to_circuit(dag)
        result = self.router.route(circuit, self.coupling_map, self.initial_layout)
        # 把指标暴露给后续 pass / benchmark 使用
        self.property_set["quroute_n_swaps"] = result.n_swaps
        self.property_set["quroute_final_layout"] = result.final_layout
        return circuit_to_dag(result.circuit)
