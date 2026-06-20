"""Stage-A REINFORCE trainer: a masked-MLP policy over device edges.

This is the minimal learning baseline that the GNN (Stage B) will later replace at the
*encoder* level only. The policy maps the engineered observation to a distribution over
SWAP edges, with invalid edges masked out. Trained with REINFORCE + a moving-average
baseline for variance reduction.

Requires torch (`pip install -e ".[learn]"`).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..env import RoutingEnv
from .circuits import random_circuit

NEG_INF = -1e9


class PolicyNet(nn.Module):
    def __init__(self, obs_size: int, num_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def _masked_dist(logits: torch.Tensor, mask: np.ndarray) -> torch.distributions.Categorical:
    m = torch.as_tensor(mask, dtype=torch.bool)
    masked = torch.where(m, logits, torch.full_like(logits, NEG_INF))
    return torch.distributions.Categorical(logits=masked)


class TorchMaskedPolicy:
    """Inference wrapper: (obs, info) -> action. Plugs into PolicyRouter."""

    def __init__(self, net: PolicyNet, greedy: bool = True):
        self.net = net
        self.greedy = greedy
        self._last = None  # anti-livelock: don't immediately undo the previous SWAP

    def reset(self) -> None:
        self._last = None

    @torch.no_grad()
    def __call__(self, obs: np.ndarray, info: dict) -> int:
        logits = self.net(torch.as_tensor(obs, dtype=torch.float32))
        dist = _masked_dist(logits, info["action_mask"])
        if not self.greedy:
            a = int(dist.sample())
        else:
            order = torch.argsort(dist.probs, descending=True).tolist()
            a = next((i for i in order if i != self._last and info["action_mask"][i]), order[0])
        self._last = a
        return a


def train(
    coupling_map,
    *,
    n_qubits: int,
    n_two_qubit: int = 8,
    iterations: int = 300,
    lr: float = 3e-3,
    seed: int = 0,
    log_every: int = 50,
) -> PolicyNet:
    rng = np.random.default_rng(seed)
    sample_env = RoutingEnv(coupling_map, circuit=random_circuit(n_qubits, n_two_qubit, seed=0))
    net = PolicyNet(sample_env.observation_size, sample_env.num_actions)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    baseline = 0.0

    for it in range(iterations):
        circ = random_circuit(n_qubits, n_two_qubit, seed=int(rng.integers(1 << 30)))
        env = RoutingEnv(coupling_map, circuit=circ, max_steps=40 * coupling_map.size())
        obs, info = env.reset()
        log_probs, rewards = [], []
        terminated = len(env.front) == 0
        truncated = False
        while not (terminated or truncated):
            logits = net(torch.as_tensor(obs, dtype=torch.float32))
            dist = _masked_dist(logits, info["action_mask"])
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            obs, reward, terminated, truncated, info = env.step(int(action))
            rewards.append(reward)

        if not log_probs:  # circuit needed no routing
            continue
        ep_return = float(sum(rewards))
        baseline = 0.95 * baseline + 0.05 * ep_return
        advantage = ep_return - baseline
        loss = -(torch.stack(log_probs).sum() * advantage)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if log_every and (it % log_every == 0 or it == iterations - 1):
            print(f"iter {it:4d}  return {ep_return:7.2f}  swaps {env.n_swaps:3d}  "
                  f"baseline {baseline:7.2f}")
    return net
