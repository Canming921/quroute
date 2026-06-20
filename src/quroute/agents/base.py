"""Router interface + result container.

Every routing strategy (the greedy baseline now, the RL+GNN agent later) implements
the same `BaseRouter.route(...)` contract, so they are drop-in interchangeable inside
the Qiskit `QurouteRouter` pass and the benchmark harness.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap


@dataclass
class RoutedResult:
    """Output of a routing pass plus the metrics the rubric/benchmark cares about."""

    circuit: QuantumCircuit
    n_swaps: int
    final_layout: dict[int, int]  # logical -> physical, after routing
    extra: dict = field(default_factory=dict)

    @property
    def added_cx(self) -> int:
        """Each inserted SWAP costs 3 CNOTs on a CX-native device."""
        return 3 * self.n_swaps

    @property
    def depth(self) -> int:
        return self.circuit.depth()


class BaseRouter(ABC):
    """Common interface for all routing strategies."""

    name: str = "base"

    @abstractmethod
    def route(
        self,
        circuit: QuantumCircuit,
        coupling_map: CouplingMap,
        initial_layout: dict[int, int] | None = None,
    ) -> RoutedResult:
        """Return a hardware-valid circuit (every 2-qubit gate on a coupling edge)."""
        raise NotImplementedError
