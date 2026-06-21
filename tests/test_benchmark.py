import pytest

from quroute import grid_topology
from quroute.benchmark import (
    SabreBaselineRouter,
    benchmark,
    circuit_metrics,
    default_suite,
    to_csv,
)
from quroute.benchmark.circuits import ghz_circuit, qaoa_maxcut_circuit, qft_circuit, random_circuit


def _max_arity(circuit):
    return max((inst.operation.num_qubits for inst in circuit.data), default=0)


@pytest.mark.parametrize("gen", [
    lambda: qft_circuit(5),
    lambda: ghz_circuit(5),
    lambda: qaoa_maxcut_circuit(5, seed=0),
    lambda: random_circuit(5, 8, seed=0),
])
def test_generators_are_routable(gen):
    c = gen()
    assert _max_arity(c) <= 2          # 只含 1、2 比特门
    assert "swap" not in c.count_ops() # 输入不含 SWAP -> 开销计账干净


def test_default_suite_keys():
    suite = default_suite(6, seed=1)
    assert set(suite) == {"qft", "ghz", "qaoa", "random"}


def test_metrics_keys():
    m = circuit_metrics(qft_circuit(4))
    assert set(m) == {"cx", "depth", "depth_2q"}


def test_benchmark_rows_valid_and_complete():
    from quroute import GreedyShortestPathRouter

    cm = grid_topology(2, 3)
    suite = default_suite(6, seed=0)
    routers = {"greedy": GreedyShortestPathRouter(), "sabre": SabreBaselineRouter(trials=2)}
    rows = benchmark(cm, suite, routers, n_qubits=6)
    assert len(rows) == len(suite) * len(routers)
    assert all(r["valid"] for r in rows)              # 每个输出都硬件合法
    assert all(r["added_cx"] >= 0 for r in rows)


def test_to_csv_writes(tmp_path):
    from quroute import GreedyShortestPathRouter

    cm = grid_topology(2, 2)
    rows = benchmark(cm, default_suite(4, seed=0),
                     {"greedy": GreedyShortestPathRouter()}, n_qubits=4)
    p = tmp_path / "r.csv"
    to_csv(rows, str(p))
    assert p.exists() and p.read_text().startswith("circuit,")


def test_plots_smoke(tmp_path):
    pytest.importorskip("matplotlib")
    from quroute import GreedyShortestPathRouter
    from quroute.benchmark.plots import plot_added_cx_by_circuit

    cm = grid_topology(2, 2)
    rows = benchmark(cm, default_suite(4, seed=0),
                     {"greedy": GreedyShortestPathRouter(), "sabre": SabreBaselineRouter(2)},
                     n_qubits=4)
    out = tmp_path / "fig.png"
    plot_added_cx_by_circuit(rows, str(out), n_qubits=4)
    assert out.exists()
