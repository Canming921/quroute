"""Stage-B: a Graph Neural Network policy over the device topology.

Why a GNN (and the whole point of Stage B):
- The agent must "learn the constraint features of the target physical topology"
  (course rubric). Message passing over the coupling graph does exactly that.
- Unlike the Stage-A MLP — whose input width is tied to one device — this policy is
  *size-agnostic*: the same shared GNN + edge-MLP weights run on ANY topology, so we can
  train on one device and transfer to another with no retraining.

Implementation is a from-scratch GCN (dense symmetric-normalized propagation), so the
only dependency is torch (no torch-geometric). GINEConv / GAT are drop-in upgrades.

Requires the learn extra:  pip install -e ".[learn]"
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..env import RoutingEnv
from .circuits import random_circuit

NEG_INF = -1e9


class GCNLayer(nn.Module):
    """h' = Â · (h W).  Â is the symmetric-normalized adjacency with self-loops."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        return a_hat @ self.lin(x)


class GNNPolicyNet(nn.Module):
    def __init__(self, in_feats: int = 4, hidden: int = 64, layers: int = 3, edge_feats: int = 1):
        super().__init__()
        dims = [in_feats] + [hidden] * layers
        self.gcns = nn.ModuleList(GCNLayer(dims[i], dims[i + 1]) for i in range(layers))
        # edge score from the two endpoint embeddings (symmetric) + physical edge features
        # (the per-edge front-layer distance reduction — the signal the greedy heuristic uses)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden + edge_feats, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def node_embeddings(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        h = x
        for gcn in self.gcns:
            h = torch.relu(gcn(h, a_hat))
        return h

    def forward(self, x, a_hat, edges, edge_feats) -> torch.Tensor:
        h = self.node_embeddings(x, a_hat)
        idx = torch.as_tensor(edges, dtype=torch.long)  # [E, 2]
        ef = torch.as_tensor(edge_feats, dtype=torch.float32).reshape(len(edges), -1)
        z = torch.cat([h[idx[:, 0]] + h[idx[:, 1]], ef], dim=-1)  # symmetric endpoints + edge feat
        return self.edge_mlp(z).squeeze(-1)  # [E] logits over candidate SWAP edges


def _edge_reduction(obs: np.ndarray, n_nodes: int, n_edges: int) -> np.ndarray:
    """Per-edge front-layer distance reduction, sliced from the observation tail."""
    return obs[2 * n_nodes : 2 * n_nodes + n_edges]


def _masked_dist(logits: torch.Tensor, mask: np.ndarray) -> torch.distributions.Categorical:
    m = torch.as_tensor(mask, dtype=torch.bool)
    masked = torch.where(m, logits, torch.full_like(logits, NEG_INF))
    return torch.distributions.Categorical(logits=masked)


class GNNMaskedPolicy:
    """Inference wrapper: (obs, info) -> action. Reads the graph state from `info`,
    so it plugs into `PolicyRouter` / `run_episode` unchanged.

    Termination guarantee: if no gate has executed for `patience` steps, the policy
    forces the max-distance-reduction SWAP (available in the observation tail), which
    strictly decreases the front-layer distance and therefore makes provable progress.
    Between those, a short tabu blocks trivial 2-cycles. This holds on any connected
    topology, including low-connectivity ones (a line) the net was never trained on."""

    def __init__(self, net: GNNPolicyNet, greedy: bool = True, tabu: int = 4,
                 patience: int | None = None):
        from collections import deque

        self.net = net
        self.greedy = greedy
        self._tabu = deque(maxlen=tabu)
        self._patience = patience
        self._last_exec = 0
        self._stall = 0

    def reset(self) -> None:
        self._tabu.clear()
        self._last_exec = 0
        self._stall = 0

    def _fallback(self, obs, info, mask):
        n = info["adjacency"].shape[0]
        e = len(info["action_edges"])
        edge_red = obs[2 * n : 2 * n + e]
        return int(np.argmax(np.where(mask, edge_red, -np.inf)))

    @torch.no_grad()
    def __call__(self, obs: np.ndarray, info: dict) -> int:
        mask = info["action_mask"]
        n = info["adjacency"].shape[0]
        patience = self._patience if self._patience is not None else 2 * n

        # stall detection (forces guaranteed progress)
        if info["n_executed"] > self._last_exec:
            self._stall = 0
            self._last_exec = info["n_executed"]
        else:
            self._stall += 1
        if self._stall >= patience:
            action = self._fallback(obs, info, mask)
            self._tabu.append(action)
            return action

        x = torch.as_tensor(info["node_features"], dtype=torch.float32)
        a = torch.as_tensor(info["adjacency"], dtype=torch.float32)
        ef = _edge_reduction(obs, n, len(info["action_edges"]))
        logits = self.net(x, a, info["action_edges"], ef)
        dist = _masked_dist(logits, mask)
        if not self.greedy:
            action = int(dist.sample())
        else:
            order = torch.argsort(dist.probs, descending=True).tolist()
            action = next((i for i in order if mask[i] and i not in self._tabu), None)
            if action is None:
                action = self._fallback(obs, info, mask)
        self._tabu.append(action)
        return action


def train_gnn(
    coupling_maps,
    *,
    n_two_qubit: int = 8,
    iterations: int = 400,
    lr: float = 3e-3,
    hidden: int = 64,
    layers: int = 3,
    ent_coef: float = 0.01,
    seed: int = 0,
    log_every: int = 100,
) -> GNNPolicyNet:
    """REINFORCE (+ entropy bonus) training. Pass a LIST of coupling maps to train
    across topologies (the generalization story); a single map also works."""
    rng = np.random.default_rng(seed)
    cmaps = list(coupling_maps) if isinstance(coupling_maps, (list, tuple)) else [coupling_maps]
    net = GNNPolicyNet(in_feats=4, hidden=hidden, layers=layers)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    baseline = 0.0

    for it in range(iterations):
        cm = cmaps[int(rng.integers(len(cmaps)))]
        n = cm.size()
        circ = random_circuit(n, n_two_qubit, seed=int(rng.integers(1 << 30)))
        env = RoutingEnv(cm, circuit=circ, max_steps=40 * n)
        obs, info = env.reset()
        log_probs, rewards, entropies = [], [], []
        terminated = len(env.front) == 0
        truncated = False
        while not (terminated or truncated):
            x = torch.as_tensor(info["node_features"], dtype=torch.float32)
            a = torch.as_tensor(info["adjacency"], dtype=torch.float32)
            ef = _edge_reduction(obs, env.n_phys, len(info["action_edges"]))
            logits = net(x, a, info["action_edges"], ef)
            dist = _masked_dist(logits, info["action_mask"])
            action = dist.sample()
            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
            obs, reward, terminated, truncated, info = env.step(int(action))
            rewards.append(reward)

        if not log_probs:
            continue
        ep_return = float(sum(rewards))
        baseline = 0.95 * baseline + 0.05 * ep_return
        policy_loss = -(torch.stack(log_probs).sum() * (ep_return - baseline))
        entropy_bonus = torch.stack(entropies).sum()
        loss = policy_loss - ent_coef * entropy_bonus
        opt.zero_grad()
        loss.backward()
        opt.step()

        if log_every and (it % log_every == 0 or it == iterations - 1):
            print(f"iter {it:4d}  n={n:2d}  return {ep_return:7.2f}  swaps {env.n_swaps:3d}"
                  f"  baseline {baseline:7.2f}")
    return net
