"""DAG utilities shared by every router (greedy baseline and the future RL env).

The *front layer* — the set of gates whose predecessors have all executed — is the
state the RL agent will act on. We expose it here so the agent and the baseline use
exactly the same notion of "what is executable right now".
"""
from __future__ import annotations

from qiskit.dagcircuit import DAGCircuit, DAGOpNode


def qubit_indices(dag: DAGCircuit, node: DAGOpNode) -> list[int]:
    """Logical-qubit indices touched by a DAG op node."""
    return [dag.find_bit(q).index for q in node.qargs]


def front_layer(dag: DAGCircuit) -> list[DAGOpNode]:
    """Gates currently executable (no unexecuted predecessors).

    Thin wrapper over qiskit's own front_layer() so the RL env and the baseline
    agree on terminology. Kept as a function (not inlined) because Stage B will
    extend it to also return the per-gate interaction sub-graph fed to the GNN.
    """
    return dag.front_layer()


def two_qubit_interactions(dag: DAGCircuit) -> list[tuple[int, int]]:
    """All (logical_a, logical_b) pairs that share a 2-qubit gate.

    This is the *interaction graph*; together with the coupling graph it is the
    pair of graphs the GNN encoder will ingest in Stage B.
    """
    pairs: list[tuple[int, int]] = []
    for node in dag.two_qubit_ops():
        a, b = qubit_indices(dag, node)
        pairs.append((a, b))
    return pairs
