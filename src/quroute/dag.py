"""所有路由器共用的 DAG 工具(贪心基线与未来的 RL 环境都用它)。

front layer(前沿层)—— 所有前驱都已执行的门集合 —— 正是 RL 智能体要作用其上的
状态。这里统一暴露出来,使智能体和基线对“当前哪些门可执行”使用完全一致的定义。
"""
from __future__ import annotations

from qiskit.dagcircuit import DAGCircuit, DAGOpNode


def qubit_indices(dag: DAGCircuit, node: DAGOpNode) -> list[int]:
    """某个 DAG 操作节点所作用的逻辑比特编号。"""
    return [dag.find_bit(q).index for q in node.qargs]


def front_layer(dag: DAGCircuit) -> list[DAGOpNode]:
    """当前可执行的门(没有未执行的前驱)。

    对 qiskit 自带 front_layer() 的一层薄封装,使 RL 环境与基线在术语上保持一致。
    单独写成函数(而非内联)是因为 Stage B 会扩展它,使其同时返回喂给 GNN 的、
    每个门的交互子图。
    """
    return dag.front_layer()


def two_qubit_interactions(dag: DAGCircuit) -> list[tuple[int, int]]:
    """所有共享同一个两比特门的 (逻辑比特 a, 逻辑比特 b) 对。

    这就是“交互图”;它与耦合图一起,构成 Stage B 中 GNN 编码器要读入的两张图。
    """
    pairs: list[tuple[int, int]] = []
    for node in dag.two_qubit_ops():
        a, b = qubit_indices(dag, node)
        pairs.append((a, b))
    return pairs
