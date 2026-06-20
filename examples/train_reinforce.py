"""Train the Stage-A REINFORCE policy and compare it to a random policy.

Requires the learn extra:  pip install -e ".[learn]"
Run:  python examples/train_reinforce.py
"""
import numpy as np
import torch

from quroute import RandomMaskedPolicy, RoutingEnv, grid_topology
from quroute.agents.policy_router import run_episode
from quroute.train.circuits import random_circuit
from quroute.train.reinforce import TorchMaskedPolicy, train


def main():
    torch.manual_seed(0)
    cm = grid_topology(3, 3)
    net = train(cm, n_qubits=9, n_two_qubit=8, iterations=300, seed=1)

    trained = TorchMaskedPolicy(net, greedy=True)
    rand = RandomMaskedPolicy(seed=7)
    ts, rs = [], []
    for k in range(30):
        c = random_circuit(9, 8, seed=1000 + k)
        ts.append(run_episode(RoutingEnv(cm, circuit=c, max_steps=4000), trained).n_swaps)
        rs.append(run_episode(RoutingEnv(cm, circuit=c, max_steps=4000), rand).n_swaps)
    print(
        f"\nmean swaps over 30 fresh circuits:  "
        f"trained={np.mean(ts):.2f}  random={np.mean(rs):.2f}"
    )


if __name__ == "__main__":
    main()
