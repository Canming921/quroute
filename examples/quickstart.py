"""端到端演示:三种路由器 vs Qiskit SABRE。运行:python examples/quickstart.py"""
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import SabreSwap

from quroute import (
    GreedyDistancePolicy,
    GreedyShortestPathRouter,
    PolicyRouter,
    QurouteRouter,
    RoutingEnv,
    grid_topology,
)


def make_circuit():
    qc = QuantumCircuit(9)
    qc.h(range(9))
    for a, b in [(0, 8), (2, 6), (1, 7), (0, 4), (3, 5), (2, 8)]:
        qc.cx(a, b)
    return qc


def run(router, cm):
    pm = PassManager([QurouteRouter(cm, router)])
    out = pm.run(make_circuit())
    return pm.property_set["quroute_n_swaps"], out.depth()


def main():
    cm = grid_topology(3, 3)
    dims = RoutingEnv(cm, circuit=make_circuit())
    front_greedy = PolicyRouter(GreedyDistancePolicy(dims.n_phys, dims.num_actions, seed=0))

    print(f"{'router':32}{'swaps':>7}{'depth':>7}")
    s, d = run(GreedyShortestPathRouter(), cm)
    print(f"{'topological greedy (Stage 0)':32}{s:>7}{d:>7}")
    s, d = run(front_greedy, cm)
    print(f"{'front-layer greedy (Stage A)':32}{s:>7}{d:>7}")

    best = min(
        (PassManager([SabreSwap(cm, seed=k)]).run(make_circuit()) for k in range(8)),
        key=lambda o: o.count_ops().get("swap", 0),
    )
    print(f"{'SABRE (best of 8 seeds)':32}{best.count_ops().get('swap', 0):>7}{best.depth():>7}")
    print("\n训练一个学习型策略:  python examples/train_reinforce.py")


if __name__ == "__main__":
    main()
