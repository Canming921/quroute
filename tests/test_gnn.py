"""Stage-B GNN tests. Auto-skipped when torch is absent (e.g. CI core)."""
import pytest

torch = pytest.importorskip("torch")  # noqa: F841

from quroute import grid_topology, linear_topology, ring_topology  # noqa: E402
from quroute.agents.policy_router import run_episode  # noqa: E402
from quroute.benchmark.circuits import random_circuit  # noqa: E402
from quroute.env import RoutingEnv  # noqa: E402
from quroute.train.gnn import GNNMaskedPolicy, train_gnn  # noqa: E402


def _hw_valid(circuit, cm):
    edges = {tuple(e) for e in cm.get_edges()}
    for inst in circuit.data:
        qs = [circuit.find_bit(q).index for q in inst.qubits]
        if len(qs) == 2 and (qs[0], qs[1]) not in edges and (qs[1], qs[0]) not in edges:
            return False
    return True


def test_gnn_trains_and_routes():
    cm = grid_topology(2, 3)
    net = train_gnn([cm], n_two_qubit=5, iterations=8, log_every=0, seed=0)
    policy = GNNMaskedPolicy(net, greedy=True)
    res = run_episode(RoutingEnv(cm, circuit=random_circuit(6, 6, seed=1), max_steps=4000), policy)
    assert _hw_valid(res.circuit, cm)


def test_gnn_generalizes_to_unseen_topology():
    """A net trained ONLY on a grid must still route validly on a ring/line it
    never saw — the size-agnostic property the MLP cannot have."""
    net = train_gnn([grid_topology(2, 3)], n_two_qubit=5, iterations=8, log_every=0, seed=0)
    policy = GNNMaskedPolicy(net, greedy=True)
    for cm in (ring_topology(7), linear_topology(8)):
        res = run_episode(
            RoutingEnv(cm, circuit=random_circuit(cm.size(), 6, seed=2), max_steps=8000), policy
        )
        assert _hw_valid(res.circuit, cm)  # valid + terminated (no truncation)
