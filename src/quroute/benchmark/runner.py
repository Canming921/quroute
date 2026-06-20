"""Benchmark runner + metrics. qiskit + numpy only (CI-safe)."""
from __future__ import annotations

import csv
import time
from collections import defaultdict

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import SabreSwap

from ..agents.base import BaseRouter, RoutedResult

_METRIC_BASIS = ["cx", "rz", "sx", "x", "h"]


def circuit_metrics(circuit: QuantumCircuit) -> dict:
    """Flatten SWAPs to CX, then measure on a common basis for fair comparison."""
    flat = transpile(circuit, basis_gates=_METRIC_BASIS, optimization_level=0)
    return {
        "cx": flat.count_ops().get("cx", 0),
        "depth": flat.depth(),
        "depth_2q": flat.depth(lambda inst: inst.operation.num_qubits == 2),
    }


def _is_hw_valid(circuit: QuantumCircuit, coupling_map: CouplingMap) -> bool:
    edges = {tuple(e) for e in coupling_map.get_edges()}
    for inst in circuit.data:
        qs = [circuit.find_bit(q).index for q in inst.qubits]
        if len(qs) == 2 and (qs[0], qs[1]) not in edges and (qs[1], qs[0]) not in edges:
            return False
    return True


class SabreBaselineRouter(BaseRouter):
    """Qiskit SABRE routing (best of `trials` seeds), wrapped as a BaseRouter for a
    fair, uniform comparison (routing only, from a trivial initial layout)."""

    name = "qiskit_sabre"

    def __init__(self, trials: int = 5):
        self.trials = trials

    def route(self, circuit, coupling_map, initial_layout=None) -> RoutedResult:
        best = None
        for seed in range(self.trials):
            out = PassManager([SabreSwap(coupling_map, seed=seed)]).run(circuit)
            swaps = out.count_ops().get("swap", 0)
            if best is None or swaps < best[0]:
                best = (swaps, out)
        return RoutedResult(circuit=best[1], n_swaps=best[0], final_layout={},
                            extra={"router": self.name})


def benchmark(
    coupling_map: CouplingMap,
    suite: dict[str, QuantumCircuit],
    routers: dict[str, BaseRouter],
    *,
    n_qubits: int | None = None,
) -> list[dict]:
    """Route every circuit with every router; return one row of metrics per pair."""
    rows: list[dict] = []
    for cname, circ in suite.items():
        base = circuit_metrics(circ)  # un-routed reference
        for rname, router in routers.items():
            t0 = time.perf_counter()
            res = router.route(circ, coupling_map)
            dt = time.perf_counter() - t0
            m = circuit_metrics(res.circuit)
            rows.append({
                "circuit": cname,
                "n_qubits": n_qubits if n_qubits is not None else circ.num_qubits,
                "router": rname,
                "base_cx": base["cx"],
                "cx": m["cx"],
                "added_cx": m["cx"] - base["cx"],
                "depth": m["depth"],
                "depth_2q": m["depth_2q"],
                "runtime_s": round(dt, 4),
                "valid": _is_hw_valid(res.circuit, coupling_map),
            })
    return rows


def summarize(rows: list[dict]) -> str:
    """Plain-text table: added CX and 2q-depth per (circuit, router)."""
    header = f"{'circuit':10}{'router':22}{'added_cx':>9}{'2q_depth':>9}{'valid':>7}"
    lines = [header, "-" * len(header)]
    for r in sorted(rows, key=lambda x: (x["circuit"], x["router"])):
        lines.append(
            f"{r['circuit']:10}{r['router']:22}{r['added_cx']:>9}{r['depth_2q']:>9}"
            f"{str(r['valid']):>7}"
        )
    return "\n".join(lines)


def aggregate(rows: list[dict]) -> list[dict]:
    """Mean over repeated instances, keyed by (n_qubits, circuit, router)."""
    buckets = defaultdict(list)
    for r in rows:
        buckets[(r["n_qubits"], r["circuit"], r["router"])].append(r)
    out = []
    for (nq, c, rt), rs in sorted(buckets.items()):
        out.append({
            "n_qubits": nq, "circuit": c, "router": rt, "n": len(rs),
            "added_cx": sum(x["added_cx"] for x in rs) / len(rs),
            "depth_2q": sum(x["depth_2q"] for x in rs) / len(rs),
            "runtime_s": sum(x["runtime_s"] for x in rs) / len(rs),
        })
    return out


def to_csv(rows: list[dict], path: str) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
