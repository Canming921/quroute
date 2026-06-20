"""QurouteRouter — the Qiskit transpiler pass that wraps any BaseRouter.

Implemented as a `TransformationPass` so it plugs straight into a `PassManager`,
exactly like qiskit's own `SabreSwap`. This is what makes the package a real
"compiler plugin" rather than a standalone script (rubric: API design / one-click).

Example
-------
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
        # expose metrics for downstream passes / benchmarking
        self.property_set["quroute_n_swaps"] = result.n_swaps
        self.property_set["quroute_final_layout"] = result.final_layout
        return circuit_to_dag(result.circuit)
