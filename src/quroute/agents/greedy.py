"""GreedyShortestPathRouter — the Stage-A baseline.

Naive, deterministic, and *correct*: walk the circuit in a valid topological order;
whenever a 2-qubit gate's qubits are not adjacent on the device, SWAP the first one
step-by-step along a shortest path until the two are neighbours, then emit the gate.

This is intentionally simpler than SABRE (no look-ahead, no decay heuristic, no
front-layer choice). It exists so the whole pipeline runs end-to-end from commit #1,
and so the RL+GNN agent has an honest baseline to beat. The learned agent will keep
this exact interface and replace only the *which-SWAP* decision.
"""
from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from ..topology import trivial_layout
from .base import BaseRouter, RoutedResult


class GreedyShortestPathRouter(BaseRouter):
    name = "greedy_shortest_path"

    def route(
        self,
        circuit: QuantumCircuit,
        coupling_map: CouplingMap,
        initial_layout: dict[int, int] | None = None,
    ) -> RoutedResult:
        n_phys = coupling_map.size()
        n_log = circuit.num_qubits
        if n_log > n_phys:
            raise ValueError(
                f"circuit needs {n_log} qubits but device has only {n_phys}"
            )

        log_to_phys = dict(initial_layout or trivial_layout(n_log))
        phys_to_log: dict[int, int | None] = {p: None for p in range(n_phys)}
        for log, phys in log_to_phys.items():
            phys_to_log[phys] = log

        out = QuantumCircuit(n_phys, circuit.num_clbits)
        n_swaps = 0

        def apply_swap(a: int, b: int) -> None:
            nonlocal n_swaps
            out.swap(a, b)
            n_swaps += 1
            la, lb = phys_to_log[a], phys_to_log[b]
            phys_to_log[a], phys_to_log[b] = lb, la
            if la is not None:
                log_to_phys[la] = b
            if lb is not None:
                log_to_phys[lb] = a

        for inst in circuit.data:
            op = inst.operation
            qubits = [circuit.find_bit(q).index for q in inst.qubits]
            clbits = [circuit.find_bit(c).index for c in inst.clbits]

            if len(qubits) == 1:
                out.append(op, [log_to_phys[qubits[0]]], clbits)
            elif len(qubits) == 2:
                l1, l2 = qubits
                p1, p2 = log_to_phys[l1], log_to_phys[l2]
                if coupling_map.distance(p1, p2) > 1:
                    path = coupling_map.shortest_undirected_path(p1, p2)
                    # move the logical at p1 toward p2 until adjacent (len(path)-2 swaps)
                    for k in range(len(path) - 2):
                        apply_swap(path[k], path[k + 1])
                out.append(op, [log_to_phys[l1], log_to_phys[l2]], clbits)
            else:
                raise ValueError(
                    f"{op.name} acts on {len(qubits)} qubits; decompose to <=2q first"
                )

        return RoutedResult(
            circuit=out,
            n_swaps=n_swaps,
            final_layout=dict(log_to_phys),
            extra={"router": self.name},
        )
