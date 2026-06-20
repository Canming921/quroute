"""RoutingEnv — the Stage-A reinforcement-learning environment.

A gym-style MDP for qubit routing. It is a faithful little routing *simulator*:
at every decision point the front layer contains only 2-qubit gates whose qubits are
NOT yet adjacent on the device (1-qubit gates and already-adjacent gates are executed
"for free"), so the agent's only job is to choose a SWAP.

Deliberately depends on numpy + qiskit only (no torch / gymnasium), so it runs in CI.
A trained neural policy plugs in later via `agents.PolicyRouter` without touching this
file — that is the whole point of the interface.

MDP
---
state      : engineered feature vector (see `_observation`); Stage B will feed the raw
             (coupling graph, interaction graph) to a GNN instead.
action     : index into `self.edges` — the device edge to SWAP.
mask       : `valid_action_mask()` — edges incident to a front-layer qubit (SABRE's
             candidate set). Invalid actions are still physically legal but useless.
reward     : swap_cost (default -1, so maximizing return == minimizing SWAPs)
             + shaping * (front-layer distance reduction)   [potential-based]
             + exec_bonus * (#gates executed this step).
terminated : all gates routed.   truncated : step budget exhausted.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from .agents.base import RoutedResult
from .topology import trivial_layout


class RoutingEnv:
    def __init__(
        self,
        coupling_map: CouplingMap,
        circuit: QuantumCircuit | None = None,
        circuit_factory: Callable[[], QuantumCircuit] | None = None,
        *,
        swap_cost: float = -1.0,
        shaping: float = 0.2,
        exec_bonus: float = 0.0,
        max_steps: int | None = None,
        seed: int | None = None,
    ):
        if circuit is None and circuit_factory is None:
            raise ValueError("provide either `circuit` or `circuit_factory`")
        self.coupling_map = coupling_map
        self._fixed_circuit = circuit
        self._circuit_factory = circuit_factory
        self.swap_cost = swap_cost
        self.shaping = shaping
        self.exec_bonus = exec_bonus
        self.n_phys = coupling_map.size()

        # undirected, de-duplicated edge list -> the action space
        edges = set()
        for a, b in coupling_map.get_edges():
            edges.add((a, b) if a < b else (b, a))
        self.edges: list[tuple[int, int]] = sorted(edges)
        self.num_actions = len(self.edges)

        self._dist = np.asarray(coupling_map.distance_matrix, dtype=int)
        self.max_steps = max_steps if max_steps is not None else 50 * self.n_phys
        self.rng = np.random.default_rng(seed)

        self._build_graph_caches()
        self._reset_internal_state()

    # ---- static graph caches (for the Stage-B GNN) ------------------------
    def _build_graph_caches(self) -> None:
        n = self.n_phys
        a = np.zeros((n, n), dtype=np.float64)
        for u, v in self.edges:
            a[u, v] = 1.0
            a[v, u] = 1.0
        self._deg = a.sum(1)
        self._maxdeg = max(1.0, float(self._deg.max()))
        # symmetric-normalized adjacency with self-loops: D^-1/2 (A+I) D^-1/2
        ah = a + np.eye(n)
        dinv = 1.0 / np.sqrt(ah.sum(1))
        self._norm_adj = (((ah * dinv).T) * dinv).astype(np.float32)
        self._maxdist = max(1, int(self._dist.max()))

    def node_features(self) -> np.ndarray:
        """Per-physical-qubit features the GNN reads (topology-agnostic, fixed width=4).

        [occupied, in_front_layer, distance_to_gate_partner (norm), degree (norm)].
        Each occupied qubit has at most one front-layer gate (per-qubit dependency
        chain), so its partner is well-defined.
        """
        n = self.n_phys
        feats = np.zeros((n, 4), dtype=np.float32)
        partner_phys: dict[int, int] = {}
        for gid in self.front:
            g = self.gates[gid]
            if g["is2q"]:
                p0, p1 = (self.log_to_phys[q] for q in g["qubits"])
                partner_phys[p0] = p1
                partner_phys[p1] = p0
        for p in range(n):
            if self.phys_to_log[p] is not None:
                feats[p, 0] = 1.0
            if p in partner_phys:
                feats[p, 1] = 1.0
                feats[p, 2] = self._dist[p, partner_phys[p]] / self._maxdist
            feats[p, 3] = self._deg[p] / self._maxdeg
        return feats

    # ---- circuit -> gate dependency graph ---------------------------------
    def _build_gate_graph(self, circuit: QuantumCircuit):
        gates = []
        preds = defaultdict(set)
        succs = defaultdict(set)
        last_on_qubit: dict[int, int] = {}
        last_on_clbit: dict[int, int] = {}
        for inst in circuit.data:
            gid = len(gates)
            qubits = [circuit.find_bit(q).index for q in inst.qubits]
            clbits = [circuit.find_bit(c).index for c in inst.clbits]
            if len(qubits) > 2:
                raise ValueError(f"{inst.operation.name} has >2 qubits; decompose first")
            for q in qubits:
                if q in last_on_qubit:
                    p = last_on_qubit[q]
                    preds[gid].add(p)
                    succs[p].add(gid)
                last_on_qubit[q] = gid
            for c in clbits:
                if c in last_on_clbit:
                    p = last_on_clbit[c]
                    preds[gid].add(p)
                    succs[p].add(gid)
                last_on_clbit[c] = gid
            gates.append(
                {"op": inst.operation, "qubits": qubits, "clbits": clbits,
                 "is2q": len(qubits) == 2}
            )
        return gates, preds, succs

    def _reset_internal_state(self):
        circuit = self._fixed_circuit
        if circuit is None:
            circuit = self._circuit_factory()
        self._circuit = circuit
        if circuit.num_qubits > self.n_phys:
            raise ValueError("circuit needs more qubits than the device has")
        self.gates, self._preds, self._succs = self._build_gate_graph(circuit)
        self._required = {gid: len(self._preds[gid]) for gid in range(len(self.gates))}
        self.front: list[int] = [g for g in range(len(self.gates)) if self._required[g] == 0]
        self.log_to_phys = dict(trivial_layout(circuit.num_qubits))
        self.phys_to_log: dict[int, int | None] = {p: None for p in range(self.n_phys)}
        for log, phys in self.log_to_phys.items():
            self.phys_to_log[phys] = log
        self.out = QuantumCircuit(self.n_phys, circuit.num_clbits)
        self.n_swaps = 0
        self.steps = 0
        self.n_executed = 0

    # ---- core mechanics ---------------------------------------------------
    def _is_executable(self, gid: int) -> bool:
        g = self.gates[gid]
        if not g["is2q"]:
            return True
        p0, p1 = (self.log_to_phys[q] for q in g["qubits"])
        return self._dist[p0, p1] == 1

    def _execute_free_gates(self) -> int:
        """Greedily emit every currently-executable front-layer gate. Returns count."""
        executed = 0
        changed = True
        while changed:
            changed = False
            for gid in list(self.front):
                if self._is_executable(gid):
                    g = self.gates[gid]
                    phys = [self.log_to_phys[q] for q in g["qubits"]]
                    self.out.append(g["op"], phys, g["clbits"])
                    self.front.remove(gid)
                    executed += 1
                    self.n_executed += 1
                    for s in self._succs[gid]:
                        self._required[s] -= 1
                        if self._required[s] == 0:
                            self.front.append(s)
                    changed = True
        return executed

    def _apply_swap(self, a: int, b: int) -> None:
        self.out.swap(a, b)
        self.n_swaps += 1
        la, lb = self.phys_to_log[a], self.phys_to_log[b]
        self.phys_to_log[a], self.phys_to_log[b] = lb, la
        if la is not None:
            self.log_to_phys[la] = b
        if lb is not None:
            self.log_to_phys[lb] = a

    def _front_distance_sum(self) -> int:
        total = 0
        for gid in self.front:
            g = self.gates[gid]
            if g["is2q"]:
                p0, p1 = (self.log_to_phys[q] for q in g["qubits"])
                total += int(self._dist[p0, p1])
        return total

    # ---- gym-style API ----------------------------------------------------
    def reset(self, *, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_internal_state()
        self._execute_free_gates()  # nothing to decide until we're stuck
        return self._observation(), self._info()

    def step(self, action: int):
        if not (0 <= action < self.num_actions):
            raise IndexError(f"action {action} out of range [0,{self.num_actions})")
        self.steps += 1
        dist_before = self._front_distance_sum()
        a, b = self.edges[action]
        self._apply_swap(a, b)
        executed = self._execute_free_gates()
        dist_after = self._front_distance_sum()

        reward = (
            self.swap_cost
            + self.shaping * (dist_before - dist_after)
            + self.exec_bonus * executed
        )
        terminated = len(self.front) == 0
        truncated = (not terminated) and self.steps >= self.max_steps
        return self._observation(), float(reward), terminated, truncated, self._info()

    # ---- observation / mask ----------------------------------------------
    def valid_action_mask(self) -> np.ndarray:
        """Edges incident to a physical qubit holding a front-layer (2q) qubit."""
        hot = np.zeros(self.n_phys, dtype=bool)
        for gid in self.front:
            g = self.gates[gid]
            if g["is2q"]:
                for q in g["qubits"]:
                    hot[self.log_to_phys[q]] = True
        mask = np.array([hot[a] or hot[b] for a, b in self.edges], dtype=bool)
        if not mask.any():  # safety: never hand back an all-False mask
            mask[:] = True
        return mask

    def _edge_distance_reduction(self) -> np.ndarray:
        """Per-edge: how much would swapping it reduce the front-layer distance sum."""
        red = np.zeros(self.num_actions, dtype=float)
        base = self._front_distance_sum()
        for i, (a, b) in enumerate(self.edges):
            la, lb = self.phys_to_log[a], self.phys_to_log[b]
            # simulate the swap cheaply on the layout
            self.phys_to_log[a], self.phys_to_log[b] = lb, la
            if la is not None:
                self.log_to_phys[la] = b
            if lb is not None:
                self.log_to_phys[lb] = a
            red[i] = base - self._front_distance_sum()
            # undo
            self.phys_to_log[a], self.phys_to_log[b] = la, lb
            if la is not None:
                self.log_to_phys[la] = a
            if lb is not None:
                self.log_to_phys[lb] = b
        return red

    def _observation(self) -> np.ndarray:
        n = self.n_phys
        occ = np.zeros(n, dtype=float)
        front_hot = np.zeros(n, dtype=float)
        for p in range(n):
            log = self.phys_to_log[p]
            if log is not None:
                occ[p] = (log + 1) / n
        for gid in self.front:
            g = self.gates[gid]
            if g["is2q"]:
                for q in g["qubits"]:
                    front_hot[self.log_to_phys[q]] = 1.0
        edge_red = self._edge_distance_reduction()
        if edge_red.size and np.abs(edge_red).max() > 0:
            edge_red = edge_red / np.abs(edge_red).max()
        progress = np.array([self.n_executed / max(1, len(self.gates))], dtype=float)
        return np.concatenate([occ, front_hot, edge_red, progress]).astype(np.float32)

    @property
    def observation_size(self) -> int:
        return 2 * self.n_phys + self.num_actions + 1

    def _info(self) -> dict:
        return {
            "action_mask": self.valid_action_mask(),
            "n_swaps": self.n_swaps,
            "n_executed": self.n_executed,
            "front_size": len(self.front),
            # graph state for the Stage-B GNN policy:
            "node_features": self.node_features(),
            "adjacency": self._norm_adj,
            "action_edges": self.edges,
        }

    def result(self) -> RoutedResult:
        return RoutedResult(
            circuit=self.out,
            n_swaps=self.n_swaps,
            final_layout=dict(self.log_to_phys),
            extra={"router": "rl_env", "steps": self.steps},
        )
