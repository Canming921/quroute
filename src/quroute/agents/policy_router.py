"""PolicyRouter —— 在 RoutingEnv 上运行一个策略,并返回 RoutedResult。

这一步闭环:任意策略(现在是脚本式的,以后是训练好的 RL+GNN 网络)都能变成一个
可在 Qiskit `QurouteRouter` pass 中使用的、即插即用的 `BaseRouter`。Stage B 的
`LearnedRouter` 本质上就是 `PolicyRouter` 包了一个训练好的策略。
"""
from __future__ import annotations

from collections.abc import Callable

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from ..env import RoutingEnv
from .base import BaseRouter, RoutedResult

Policy = Callable[..., int]  # (obs, info) -> action


def run_episode(env: RoutingEnv, policy: Policy, *, seed: int | None = None) -> RoutedResult:
    obs, info = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()
    terminated = len(env.front) == 0
    truncated = False
    while not (terminated or truncated):
        action = policy(obs, info)
        obs, _reward, terminated, truncated, info = env.step(action)
    if truncated:
        raise RuntimeError(
            "episode truncated before routing finished (step budget exhausted); "
            "policy may be stuck — increase max_steps or check the policy"
        )
    return env.result()


class PolicyRouter(BaseRouter):
    name = "policy_router"

    def __init__(self, policy: Policy, *, max_steps: int | None = None, seed: int | None = None):
        self.policy = policy
        self.max_steps = max_steps
        self.seed = seed

    def route(
        self,
        circuit: QuantumCircuit,
        coupling_map: CouplingMap,
        initial_layout: dict[int, int] | None = None,
    ) -> RoutedResult:
        env = RoutingEnv(coupling_map, circuit=circuit, max_steps=self.max_steps)
        return run_episode(env, self.policy, seed=self.seed)
