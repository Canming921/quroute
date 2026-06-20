import pytest
from qiskit import QuantumCircuit

from quroute import (
    GreedyShortestPathRouter,
    LearnedRouter,
    QurouteRouter,
    grid_topology,
    linear_topology,
)


def _all_two_qubit_gates_are_legal(circuit, coupling_map):
    """Every 2-qubit gate must sit on an edge of the device."""
    edges = {tuple(e) for e in coupling_map.get_edges()}
    for inst in circuit.data:
        qs = [circuit.find_bit(q).index for q in inst.qubits]
        if len(qs) == 2:
            if (qs[0], qs[1]) not in edges and (qs[1], qs[0]) not in edges:
                return False
    return True


def test_greedy_produces_hardware_valid_circuit():
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 3)   # not adjacent on a line -> must insert SWAPs
    qc.cx(1, 3)
    qc.cx(0, 2)
    cm = linear_topology(4)
    res = GreedyShortestPathRouter().route(qc, cm)
    assert res.n_swaps > 0
    assert _all_two_qubit_gates_are_legal(res.circuit, cm)


def test_greedy_no_swaps_when_already_local():
    qc = QuantumCircuit(3)
    qc.cx(0, 1)
    qc.cx(1, 2)
    cm = linear_topology(3)
    res = GreedyShortestPathRouter().route(qc, cm)
    assert res.n_swaps == 0


def test_pass_integration():
    from qiskit.transpiler import PassManager

    qc = QuantumCircuit(6)
    qc.cx(0, 5)
    qc.cx(1, 4)
    cm = grid_topology(2, 3)
    pm = PassManager([QurouteRouter(cm, GreedyShortestPathRouter())])
    routed = pm.run(qc)
    assert _all_two_qubit_gates_are_legal(routed, cm)


def test_added_cx_accounting():
    res = GreedyShortestPathRouter().route(
        _line_circuit(), linear_topology(4)
    )
    assert res.added_cx == 3 * res.n_swaps


def _line_circuit():
    qc = QuantumCircuit(4)
    qc.cx(0, 3)
    return qc


def test_learned_router_is_stubbed():
    with pytest.raises(NotImplementedError):
        LearnedRouter().route(QuantumCircuit(2), linear_topology(2))
