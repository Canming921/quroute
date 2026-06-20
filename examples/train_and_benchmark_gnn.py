"""Stage B: train the GNN policy, benchmark it vs SABRE, and show zero-shot transfer.

Requires the learn extra:  pip install -e ".[learn,dev]"
Run:  python examples/train_and_benchmark_gnn.py
"""
import numpy as np
import torch

from quroute import (
    GreedyDistancePolicy,
    GreedyShortestPathRouter,
    LearnedRouter,
    PolicyRouter,
    RoutingEnv,
    grid_topology,
    linear_topology,
    ring_topology,
)
from quroute.agents.policy_router import run_episode
from quroute.benchmark import SabreBaselineRouter, benchmark, default_suite
from quroute.benchmark.circuits import random_circuit
from quroute.train.gnn import GNNMaskedPolicy, train_gnn


def _greedy(cm):
    e = len(RoutingEnv(cm, circuit=random_circuit(cm.size(), 8, seed=0)).edges)
    return GreedyDistancePolicy(cm.size(), e, seed=0)


def _mean_swaps(cm, policy, k=25):
    return np.mean([
        run_episode(
            RoutingEnv(cm, circuit=random_circuit(cm.size(), 8, seed=5000 + i), max_steps=8000),
            policy,
        ).n_swaps
        for i in range(k)
    ])


def main():
    torch.manual_seed(0)
    train_maps = [grid_topology(3, 3), grid_topology(2, 3), ring_topology(6), linear_topology(6)]
    print("training GNN on mixed topologies (3x3, 2x3, ring-6, line-6)...")
    net = train_gnn(
        train_maps, n_two_qubit=8, iterations=1200, ent_coef=0.02, seed=1, log_every=400
    )
    gnn = GNNMaskedPolicy(net, greedy=True)
    torch.save(net.state_dict(), "gnn_policy.pt")
    print("saved trained model -> gnn_policy.pt")

    # 1) benchmark on the 3x3 grid vs SABRE
    cm = grid_topology(3, 3)
    routers = {
        "topological_greedy": GreedyShortestPathRouter(),
        "front_layer_greedy": PolicyRouter(_greedy(cm), max_steps=8000),
        "gnn_rl (ours)": LearnedRouter(gnn, max_steps=8000),
        "qiskit_sabre(best5)": SabreBaselineRouter(trials=5),
    }
    rows = []
    for inst in range(3):
        rows += benchmark(cm, default_suite(9, seed=100 + inst), routers, n_qubits=9)
    print("\n=== benchmark on 3x3 grid (mean of 3 instances) ===")
    from quroute.benchmark.runner import aggregate
    print(f"{'circuit':10}{'router':22}{'added_cx':>9}{'2q_depth':>9}")
    for a in sorted(aggregate(rows), key=lambda x: (x["circuit"], x["router"])):
        print(f"{a['circuit']:10}{a['router']:22}{a['added_cx']:>9.1f}{a['depth_2q']:>9.1f}")

    # 2) zero-shot transfer to topologies never trained on
    print("\n=== zero-shot transfer (same net, UNSEEN topologies) ===")
    print(f"{'topology':18}{'GNN':>7}{'greedy':>8}{'random':>8}")
    from quroute import RandomMaskedPolicy
    for name, tcm in [("4x4 grid", grid_topology(4, 4)), ("ring-8", ring_topology(8)),
                      ("line-8", linear_topology(8))]:
        g = _mean_swaps(tcm, gnn)
        gr = _mean_swaps(tcm, _greedy(tcm))
        rd = _mean_swaps(tcm, RandomMaskedPolicy(seed=0))
        print(f"{name:18}{g:>7.2f}{gr:>8.2f}{rd:>8.2f}")
    print("\nThe MLP policy cannot even run these — its input size is tied to one device.")


if __name__ == "__main__":
    main()
