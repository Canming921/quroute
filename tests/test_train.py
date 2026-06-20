"""Stage-A training smoke test. Skipped automatically when torch is absent (e.g. CI)."""
import pytest

torch = pytest.importorskip("torch")  # noqa: F841

from quroute import RoutingEnv, grid_topology  # noqa: E402
from quroute.agents.policy_router import run_episode  # noqa: E402
from quroute.train.circuits import random_circuit  # noqa: E402
from quroute.train.reinforce import TorchMaskedPolicy, train  # noqa: E402


def _hw_valid(circuit, cm):
    edges = {tuple(e) for e in cm.get_edges()}
    for inst in circuit.data:
        qs = [circuit.find_bit(q).index for q in inst.qubits]
        if len(qs) == 2 and (qs[0], qs[1]) not in edges and (qs[1], qs[0]) not in edges:
            return False
    return True


def test_train_runs_and_policy_routes():
    cm = grid_topology(2, 3)
    net = train(cm, n_qubits=6, n_two_qubit=5, iterations=5, log_every=0, seed=0)
    policy = TorchMaskedPolicy(net, greedy=True)
    res = run_episode(RoutingEnv(cm, circuit=random_circuit(6, 5, seed=99), max_steps=2000), policy)
    assert _hw_valid(res.circuit, cm)


def test_random_circuit_respects_qubit_count():
    qc = random_circuit(5, 4, seed=1)
    assert qc.num_qubits == 5
