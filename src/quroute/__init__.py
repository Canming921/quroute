"""quroute — AI-driven qubit mapping & routing for restricted quantum topologies.

A Qiskit-compatible transpiler plugin that learns the connectivity constraints of
a target physical topology and decides an initial qubit mapping + SWAP-insertion
strategy that minimizes circuit depth and CNOT count.

Roadmap (see docs/physics and the README):
    Stage A  : RL environment + simple-feature policy  (baseline: GreedyShortestPathRouter)
    Stage B  : GNN topology encoder inside the RL policy (the differentiator)
    Plan B   : imitation-learning fallback (imitate SABRE) if RL training is unstable
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
