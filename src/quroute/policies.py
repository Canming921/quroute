"""RoutingEnv 的脚本式(非学习)策略。

它们的作用:(a) 让我们在没有任何 ML 依赖的情况下对环境做单元测试;
(b) 提供一个诚实的、感知 front layer 的基线(`GreedyDistancePolicy`,即 SABRE-lite);
(c) 作为 Plan B 模仿学习的行为克隆目标。

所谓“策略”就是:observation, info -> action(整数)。
"""
from __future__ import annotations

import numpy as np


class RandomMaskedPolicy:
    """在当前合法的 SWAP 中均匀随机选择。用于自检 / 作为探索下限。"""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs: np.ndarray, info: dict) -> int:
        mask = info["action_mask"]
        valid = np.flatnonzero(mask)
        return int(self.rng.choice(valid))


class GreedyDistancePolicy:
    """选择能最大程度减小 front-layer 距离之和的合法 SWAP。

    每条边的距离削减量是 `RoutingEnv._observation` 构造的观测向量的尾部块
    (已归一化),所以该策略不需要额外访问环境。平局时随机打破,避免死循环。
    """

    def __init__(self, n_phys: int, num_actions: int, seed: int | None = None):
        self.n_phys = n_phys
        self.num_actions = num_actions
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs: np.ndarray, info: dict) -> int:
        start = 2 * self.n_phys
        edge_red = obs[start : start + self.num_actions]
        mask = info["action_mask"]
        scores = np.where(mask, edge_red, -np.inf)
        best = scores.max()
        candidates = np.flatnonzero(scores == best)
        return int(self.rng.choice(candidates))
