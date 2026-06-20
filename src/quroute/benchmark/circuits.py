"""Standard benchmark circuit generators.

Every generator returns a circuit already decomposed to <=2-qubit gates with NO swap
gates, so any SWAP in a routed output is purely routing overhead (clean accounting).
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.synthesis import synth_qft_full

# basis with only 1- and 2-qubit gates and no SWAP
_BASIS = ["cx", "rz", "sx", "x", "h"]


def _flatten(circuit: QuantumCircuit) -> QuantumCircuit:
    return transpile(circuit, basis_gates=_BASIS, optimization_level=0)


def qft_circuit(n: int) -> QuantumCircuit:
    return _flatten(synth_qft_full(n))


def ghz_circuit(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return _flatten(qc)


def qaoa_maxcut_circuit(n: int, *, p: int = 1, seed=None) -> QuantumCircuit:
    """QAOA MaxCut ansatz on a random graph (~1.5n edges). Routing depends on the
    interaction (ZZ) structure, not the angles, so angles are arbitrary fixed values."""
    rng = np.random.default_rng(seed)
    edges = set()
    target = int(1.5 * n)
    while len(edges) < target:
        a, b = rng.choice(n, size=2, replace=False)
        edges.add((min(a, b), max(a, b)))
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for _ in range(p):
        for u, v in edges:
            qc.cx(int(u), int(v))
            qc.rz(0.7, int(v))
            qc.cx(int(u), int(v))
        for q in range(n):
            qc.rx(0.4, q)
    return _flatten(qc)


def random_circuit(n: int, n_two_qubit: int, *, seed=None) -> QuantumCircuit:
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for _ in range(n_two_qubit):
        a, b = rng.choice(n, size=2, replace=False)
        qc.cx(int(a), int(b))
        if rng.random() < 0.3:
            qc.h(int(rng.integers(n)))
    return _flatten(qc)


def default_suite(n: int, *, seed=None) -> dict[str, QuantumCircuit]:
    rng = np.random.default_rng(seed)
    return {
        "qft": qft_circuit(n),
        "ghz": ghz_circuit(n),
        "qaoa": qaoa_maxcut_circuit(n, p=1, seed=int(rng.integers(1 << 30))),
        "random": random_circuit(n, n_two_qubit=2 * n, seed=int(rng.integers(1 << 30))),
    }
