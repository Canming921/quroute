"""GreedyShortestPathRouter —— Stage A 的基线路由器。

朴素、确定性、且*正确*:按合法的拓扑序遍历电路;每当某个两比特门的两个比特
在设备上不相邻时,就沿最短路径把第一个比特逐步 SWAP 过去,直到两者相邻,再
执行该门。

它刻意比 SABRE 更简单(没有 look-ahead、没有衰减启发式、不挑选 front layer)。
它的存在是为了让整条流水线从第 1 个 commit 起就端到端可用,也让 RL+GNN 智能体
有一个诚实的基线可以去超越。学习型智能体会沿用完全相同的接口,只替换“选哪个
SWAP”这一决策。
"""
from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from ..topology import trivial_layout
from .base import BaseRouter, RoutedResult


class GreedyShortestPathRouter(BaseRouter):
    name = "greedy_shortest_path"

    def route(
        self,
        circuit: QuantumCircuit,
        coupling_map: CouplingMap,
        initial_layout: dict[int, int] | None = None,
    ) -> RoutedResult:
        n_phys = coupling_map.size()
        n_log = circuit.num_qubits
        if n_log > n_phys:
            raise ValueError(
                f"circuit needs {n_log} qubits but device has only {n_phys}"
            )

        log_to_phys = dict(initial_layout or trivial_layout(n_log))
        phys_to_log: dict[int, int | None] = {p: None for p in range(n_phys)}
        for log, phys in log_to_phys.items():
            phys_to_log[phys] = log

        out = QuantumCircuit(n_phys, circuit.num_clbits)
        n_swaps = 0

        def apply_swap(a: int, b: int) -> None:
            nonlocal n_swaps
            out.swap(a, b)
            n_swaps += 1
            la, lb = phys_to_log[a], phys_to_log[b]
            phys_to_log[a], phys_to_log[b] = lb, la
            if la is not None:
                log_to_phys[la] = b
            if lb is not None:
                log_to_phys[lb] = a

        for inst in circuit.data:
            op = inst.operation
            qubits = [circuit.find_bit(q).index for q in inst.qubits]
            clbits = [circuit.find_bit(c).index for c in inst.clbits]

            if len(qubits) == 1:
                out.append(op, [log_to_phys[qubits[0]]], clbits)
            elif len(qubits) == 2:
                l1, l2 = qubits
                p1, p2 = log_to_phys[l1], log_to_phys[l2]
                if coupling_map.distance(p1, p2) > 1:
                    path = coupling_map.shortest_undirected_path(p1, p2)
                    # 把 p1 上的逻辑比特沿路径挪向 p2,直到相邻(共 len(path)-2 次 SWAP)
                    for k in range(len(path) - 2):
                        apply_swap(path[k], path[k + 1])
                out.append(op, [log_to_phys[l1], log_to_phys[l2]], clbits)
            else:
                raise ValueError(
                    f"{op.name} acts on {len(qubits)} qubits; decompose to <=2q first"
                )

        return RoutedResult(
            circuit=out,
            n_swaps=n_swaps,
            final_layout=dict(log_to_phys),
            extra={"router": self.name},
        )
