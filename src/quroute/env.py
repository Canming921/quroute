"""RoutingEnv —— Stage A 的强化学习环境。

一个用于比特路由的 gym 式 MDP。它是一个忠实的小型路由*模拟器*:在每个决策点,
front layer 里只剩下两比特门、且其两个比特在设备上还不相邻(单比特门和已相邻的
门会被“免费”执行掉),所以智能体唯一要做的就是选一个 SWAP。

它刻意只依赖 numpy + qiskit(不依赖 torch / gymnasium),因此能在 CI 中运行。训练好
的神经网络策略之后通过 `agents.PolicyRouter` 接入,无需改动本文件——这正是该接口的
意义所在。

MDP
---
状态       :工程化的特征向量(见 `_observation`);Stage B 会改成把原始的
             (耦合图, 交互图)喂给 GNN。
动作       :`self.edges` 的下标 —— 要做 SWAP 的那条设备边。
掩码       :`valid_action_mask()` —— 与 front-layer 比特相邻的边(SABRE 的候选集)。
             非法动作在物理上仍合法,但没有意义。
奖励       :swap_cost(默认 -1,于是“最大化回报”==“最小化 SWAP 数”)
             + shaping * (front-layer 距离的削减)   [势函数式整形]
             + exec_bonus * (本步执行掉的门数)。
terminated :所有门都已路由完。   truncated :步数预算耗尽。
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

        # 无向、去重后的边列表 -> 动作空间
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

    # ---- 静态图缓存(供 Stage B 的 GNN 使用)------------------------------
    def _build_graph_caches(self) -> None:
        n = self.n_phys
        a = np.zeros((n, n), dtype=np.float64)
        for u, v in self.edges:
            a[u, v] = 1.0
            a[v, u] = 1.0
        self._deg = a.sum(1)
        self._maxdeg = max(1.0, float(self._deg.max()))
        # 带自环的对称归一化邻接:D^-1/2 (A+I) D^-1/2
        ah = a + np.eye(n)
        dinv = 1.0 / np.sqrt(ah.sum(1))
        self._norm_adj = (((ah * dinv).T) * dinv).astype(np.float32)
        self._maxdist = max(1, int(self._dist.max()))

    def node_features(self) -> np.ndarray:
        """GNN 读取的每个物理比特的特征(与拓扑无关,固定宽度=4)。

        [是否被占用, 是否在 front layer, 到配对比特的距离(归一化), 度数(归一化)]。
        每个被占用的比特至多对应一个 front-layer 门(逐比特的依赖链),因此其配对
        比特是唯一确定的。
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

    # ---- 电路 -> 门依赖图 -------------------------------------------------
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

    # ---- 核心机制 ---------------------------------------------------------
    def _is_executable(self, gid: int) -> bool:
        g = self.gates[gid]
        if not g["is2q"]:
            return True
        p0, p1 = (self.log_to_phys[q] for q in g["qubits"])
        return self._dist[p0, p1] == 1

    def _execute_free_gates(self) -> int:
        """贪心地执行掉当前所有可执行的 front-layer 门。返回执行掉的门数。"""
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

    # ---- gym 式接口 -------------------------------------------------------
    def reset(self, *, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_internal_state()
        self._execute_free_gates()  # 在卡住之前没有任何需要决策的事
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

    # ---- 观测 / 掩码 ------------------------------------------------------
    def valid_action_mask(self) -> np.ndarray:
        """与“持有 front-layer(两比特)逻辑比特的物理比特”相邻的那些边。"""
        hot = np.zeros(self.n_phys, dtype=bool)
        for gid in self.front:
            g = self.gates[gid]
            if g["is2q"]:
                for q in g["qubits"]:
                    hot[self.log_to_phys[q]] = True
        mask = np.array([hot[a] or hot[b] for a, b in self.edges], dtype=bool)
        if not mask.any():  # 安全兜底:绝不返回全 False 的掩码
            mask[:] = True
        return mask

    def _edge_distance_reduction(self) -> np.ndarray:
        """逐边:若对该边做 SWAP,front-layer 距离之和会减小多少。"""
        red = np.zeros(self.num_actions, dtype=float)
        base = self._front_distance_sum()
        for i, (a, b) in enumerate(self.edges):
            la, lb = self.phys_to_log[a], self.phys_to_log[b]
            # 在映射上廉价地模拟一次该 SWAP
            self.phys_to_log[a], self.phys_to_log[b] = lb, la
            if la is not None:
                self.log_to_phys[la] = b
            if lb is not None:
                self.log_to_phys[lb] = a
            red[i] = base - self._front_distance_sum()
            # 撤销
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
            # 供 Stage B 的 GNN 策略使用的图状态:
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
