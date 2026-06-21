"""物理拓扑相关的辅助函数。

这里是对 qiskit 的 CouplingMap 的一层轻量封装,使包内其余部分
(路由器、RL 环境、GNN 编码器)对设备连通图使用统一的表示。
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

from qiskit.transpiler import CouplingMap


def _symmetric(edges: Iterable[Sequence[int]]) -> CouplingMap:
    """从边列表构建一张无向(双向)CouplingMap。

    路由关心的是“连通性”,因此把每条边都设为双向,门的方向问题留给后续的
    (基翻译)pass 处理。
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
    """一维链:0-1-2-...-(n-1)。"""
    return _symmetric((i, i + 1) for i in range(n - 1))


def ring_topology(n: int) -> CouplingMap:
    """环形:在一维链基础上再加一条 (n-1)-0 的边。"""
    edges = [(i, (i + 1) % n) for i in range(n)]
    return _symmetric(edges)


def grid_topology(rows: int, cols: int) -> CouplingMap:
    """rows x cols 的二维网格(最近邻),比特编号 = r*cols + c。"""
    cm = CouplingMap.from_grid(rows, cols)
    # from_grid 返回的已经是对称的,直接返回。
    return cm


def from_edges(edges: Iterable[Sequence[int]]) -> CouplingMap:
    """从显式边列表构建自定义拓扑(例如 heavy-hex)。"""
    return _symmetric(edges)


def trivial_layout(num_logical: int) -> dict[int, int]:
    """恒等初始映射:逻辑比特 i -> 物理比特 i。

    Stage B 会用可学习的 / VF2 风格的初始布局替换它。
    """
    return {i: i for i in range(num_logical)}
