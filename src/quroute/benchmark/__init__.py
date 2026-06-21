"""Benchmark 套件:用可复现的指标,把标准电路与 Qiskit SABRE 做对比。

核心部分(电路 + runner + 指标)只依赖 qiskit + numpy,因此能在 CI 中运行。
绘图(`plots.py`)需要 matplotlib,采用懒加载。
"""
from .circuits import (
    default_suite,
    ghz_circuit,
    qaoa_maxcut_circuit,
    qft_circuit,
    random_circuit,
)
from .runner import SabreBaselineRouter, benchmark, circuit_metrics, summarize, to_csv

__all__ = [
    "default_suite",
    "qft_circuit",
    "qaoa_maxcut_circuit",
    "random_circuit",
    "ghz_circuit",
    "benchmark",
    "circuit_metrics",
    "summarize",
    "to_csv",
    "SabreBaselineRouter",
]
