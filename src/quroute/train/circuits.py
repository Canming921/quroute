"""用作训练分布的随机电路采样器。"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit


def random_circuit(n_qubits: int, n_two_qubit: int, *, seed=None) -> QuantumCircuit:
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n_qubits)
    for _ in range(n_two_qubit):
        a, b = rng.choice(n_qubits, size=2, replace=False)
        qc.cx(int(a), int(b))
        if rng.random() < 0.3:
            qc.h(int(rng.integers(n_qubits)))
    return qc
