"""quroute —— 面向受限拓扑量子芯片的 AI 驱动比特映射与路由。

一个兼容 Qiskit 的转译器(transpiler)插件:学习目标物理拓扑的连通性约束,
自动决定初始比特映射 + SWAP 插入策略,以最小化电路深度和 CNOT 门数量。

路线图(详见 docs/physics 与 README):
    Stage A  : 强化学习环境 + 简单特征策略(基线:GreedyShortestPathRouter)
    Stage B  : 在 RL 策略中用 GNN 编码拓扑(核心差异化)
    Plan B   : 训练不稳时退回模仿学习(模仿 SABRE)
"""

from .agents.base import BaseRouter, RoutedResult
from .agents.greedy import GreedyShortestPathRouter
from .agents.learned import LearnedRouter
from .agents.policy_router import PolicyRouter, run_episode
from .env import RoutingEnv
from .pass_ import QurouteRouter
from .policies import GreedyDistancePolicy, RandomMaskedPolicy
from .topology import (
    from_edges,
    grid_topology,
    linear_topology,
    ring_topology,
    trivial_layout,
)

__version__ = "0.4.0"

__all__ = [
    "BaseRouter",
    "RoutedResult",
    "GreedyShortestPathRouter",
    "LearnedRouter",
    "PolicyRouter",
    "run_episode",
    "RoutingEnv",
    "GreedyDistancePolicy",
    "RandomMaskedPolicy",
    "QurouteRouter",
    "linear_topology",
    "ring_topology",
    "grid_topology",
    "from_edges",
    "trivial_layout",
    "__version__",
]
