"""Physical-topology helpers.

Everything here is a thin, well-documented layer over qiskit's CouplingMap so that
the rest of the package (router, RL env, GNN encoder) speaks a single representation
of the device connectivity graph.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

from qiskit.transpiler import CouplingMap


def _symmetric(edges: Iterable[Sequence[int]]) -> CouplingMap:
    """Build an undirected (bidirectional) CouplingMap from an edge list.

    Routing reasons about *connectivity*, so we make every edge bidirectional and
    leave gate-direction concerns to a later (basis-translation) pass.
    """
    cm = CouplingMap()
    seen = set()
    for a, b in edges:
        for u, v in ((a, b), (b, a)):
            if (u, v) not in seen:
                cm.add_edge(u, v)
                seen.add((u, v))
    return cm


def linear_topology(n: int) -> CouplingMap:
    """A 1-D chain: 0-1-2-...-(n-1)."""
    return _symmetric((i, i + 1) for i in range(n - 1))


def ring_topology(n: int) -> CouplingMap:
    """A ring: the linear chain plus an edge (n-1)-0."""
    edges = [(i, (i + 1) % n) for i in range(n)]
    return _symmetric(edges)


def grid_topology(rows: int, cols: int) -> CouplingMap:
    """A rows x cols 2-D grid (nearest-neighbour), qubit index = r*cols + c."""
    cm = CouplingMap.from_grid(rows, cols)
    # from_grid is already symmetric; return as-is.
    return cm


def from_edges(edges: Iterable[Sequence[int]]) -> CouplingMap:
    """Build a custom topology from an explicit edge list (e.g. heavy-hex)."""
    return _symmetric(edges)


def trivial_layout(num_logical: int) -> dict[int, int]:
    """Identity initial mapping logical_i -> physical_i.

    Stage B will replace this with a learned / VF2-style initial layout.
    """
    return {i: i for i in range(num_logical)}
