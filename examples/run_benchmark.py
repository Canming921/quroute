"""运行 benchmark 套件对比 SABRE,并输出 CSV + 图。

运行:  python examples/run_benchmark.py
(出图需要 matplotlib:pip install -e ".[dev]")
"""
from quroute import (
    GreedyDistancePolicy,
    GreedyShortestPathRouter,
    PolicyRouter,
    RoutingEnv,
    grid_topology,
)
from quroute.benchmark import SabreBaselineRouter, benchmark, default_suite, to_csv


def routers_for(cm):
    dims = RoutingEnv(cm, circuit=default_suite(cm.size())["ghz"])
    return {
        "topological_greedy": GreedyShortestPathRouter(),
        "front_layer_greedy": PolicyRouter(
            GreedyDistancePolicy(dims.n_phys, dims.num_actions, seed=0), max_steps=5000
        ),
        "qiskit_sabre(best5)": SabreBaselineRouter(trials=5),
    }


def main():
    grids = {4: (2, 2), 6: (2, 3), 9: (3, 3)}
    instances = 3
    all_rows = []
    for n, (r, c) in grids.items():
        cm = grid_topology(r, c)
        routers = routers_for(cm)
        for inst in range(instances):
            suite = default_suite(n, seed=100 + inst)
            all_rows += benchmark(cm, suite, routers, n_qubits=n)
        print(f"\n=== n={n} ({r}x{c} grid), mean of {instances} instances ===")
        from quroute.benchmark.runner import aggregate
        agg = [a for a in aggregate(all_rows) if a["n_qubits"] == n]
        print(f"{'circuit':10}{'router':22}{'added_cx':>9}{'2q_depth':>9}")
        for a in sorted(agg, key=lambda x: (x["circuit"], x["router"])):
            print(f"{a['circuit']:10}{a['router']:22}{a['added_cx']:>9.1f}{a['depth_2q']:>9.1f}")

    to_csv(all_rows, "benchmark_results.csv")
    print("\n已写出 benchmark_results.csv")
    try:
        from quroute.benchmark.plots import plot_added_cx_by_circuit, plot_scaling
        plot_added_cx_by_circuit(all_rows, "benchmark_added_cx_n9.png", n_qubits=9)
        plot_scaling(all_rows, "benchmark_scaling_random.png", circuit="random")
        print("已写出图:benchmark_added_cx_n9.png, benchmark_scaling_random.png")
    except ImportError:
        print("(安装 matplotlib 才能出图)")


if __name__ == "__main__":
    main()
