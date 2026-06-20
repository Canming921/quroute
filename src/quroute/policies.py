"""Scripted (non-learning) policies for the RoutingEnv.

These (a) let us unit-test the environment without any ML deps, (b) give an honest
front-layer-aware baseline (`GreedyDistancePolicy` ~ SABRE-lite), and (c) serve as the
behaviour-cloning target for the Plan-B imitation-learning fallback.

A "policy" is just: observation, info -> action (int).
"""
from __future__ import annotations

import numpy as np


class RandomMaskedPolicy:
    """Uniformly random over currently-valid SWAPs. Sanity-check / exploration floor."""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs: np.ndarray, info: dict) -> int:
        mask = info["action_mask"]
        valid = np.flatnonzero(mask)
        return int(self.rng.choice(valid))


class GreedyDistancePolicy:
    """Pick the valid SWAP that most reduces the front-layer distance sum.

    The per-edge distance reduction is the tail block of the observation built by
    `RoutingEnv._observation` (normalized), so the policy needs no extra env access.
    Ties broken randomly to avoid livelock.
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
