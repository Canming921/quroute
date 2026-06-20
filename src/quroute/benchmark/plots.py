"""Benchmark plots (needs matplotlib; import lazily)."""
from __future__ import annotations

from .runner import aggregate


def _import_plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError as e:  # pragma: no cover
        raise ImportError("plots require matplotlib: pip install matplotlib") from e


def plot_added_cx_by_circuit(rows: list[dict], path: str, *, n_qubits: int | None = None):
    """Grouped bar chart: added CX per circuit, one bar group per router."""
    plt = _import_plt()
    agg = aggregate(rows)
    if n_qubits is not None:
        agg = [a for a in agg if a["n_qubits"] == n_qubits]
    circuits = sorted({a["circuit"] for a in agg})
    routers = sorted({a["router"] for a in agg})
    lookup = {(a["circuit"], a["router"]): a["added_cx"] for a in agg}

    import numpy as np

    x = np.arange(len(circuits))
    w = 0.8 / max(1, len(routers))
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, rt in enumerate(routers):
        vals = [lookup.get((c, rt), 0) for c in circuits]
        ax.bar(x + i * w, vals, w, label=rt)
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(circuits)
    ax.set_ylabel("added CNOTs (lower is better)")
    title = "Routing overhead by circuit"
    if n_qubits is not None:
        title += f"  (n={n_qubits})"
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_scaling(rows: list[dict], path: str, *, circuit: str = "random"):
    """Line plot: added CX vs qubit count for one circuit family, per router."""
    plt = _import_plt()
    agg = [a for a in aggregate(rows) if a["circuit"] == circuit]
    routers = sorted({a["router"] for a in agg})
    fig, ax = plt.subplots(figsize=(8, 5))
    for rt in routers:
        pts = sorted((a["n_qubits"], a["added_cx"]) for a in agg if a["router"] == rt)
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", label=rt)
    ax.set_xlabel("number of qubits")
    ax.set_ylabel("added CNOTs (lower is better)")
    ax.set_title(f"Routing-overhead scaling — '{circuit}' circuits")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
