"""路由器接口 + 结果容器。

每一种路由策略(现在的贪心基线、之后的 RL+GNN 智能体)都实现同一个
`BaseRouter.route(...)` 契约,因此它们在 Qiskit `QurouteRouter` pass 和
benchmark 框架里可以即插即换。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap


@dataclass
class RoutedResult:
    """一次路由的输出,外加评分表 / benchmark 关心的各项指标。"""

    circuit: QuantumCircuit
    n_swaps: int
    final_layout: dict[int, int]  # 路由后的 逻辑->物理 映射
    extra: dict = field(default_factory=dict)

    @property
    def added_cx(self) -> int:
        """在以 CX 为原生门的设备上,每插入一个 SWAP 相当于 3 个 CNOT。"""
        return 3 * self.n_swaps

    @property
    def depth(self) -> int:
        return self.circuit.depth()


class BaseRouter(ABC):
    """所有路由策略的统一接口。"""

    name: str = "base"

    @abstractmethod
    def route(
        self,
        circuit: QuantumCircuit,
        coupling_map: CouplingMap,
        initial_layout: dict[int, int] | None = None,
    ) -> RoutedResult:
        """返回一个硬件合法的电路(每个两比特门都落在一条耦合边上)。"""
        raise NotImplementedError
