"""标准 benchmark 电路生成器。

每个生成器返回的电路都已分解为 ≤2 比特门、且不含 SWAP 门,因此路由输出里出现的
任何 SWAP 都纯属路由开销(便于干净地计账)。
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.synthesis import synth_qft_full

# 只含 1、2 比特门、且不含 SWAP 的基门集合
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
    """随机图(约 1.5n 条边)上的 QAOA MaxCut 拟设。路由只取决于交互(ZZ)结构,
    与角度无关,所以角度取任意固定值即可。"""
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
