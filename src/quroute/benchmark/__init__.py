"""Benchmark suite: standard circuits vs Qiskit SABRE, with reproducible metrics.

Core (circuits + runner + metrics) depends on qiskit + numpy only, so it runs in CI.
Plotting (`plots.py`) needs matplotlib and is imported lazily.
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
