# quroute

**AI-driven qubit mapping & routing for restricted quantum topologies — a Qiskit transpiler plugin.**

Quantum algorithms assume all-to-all connectivity, but real chips don't have it. Making a
two-qubit gate physically possible requires moving qubit states around with `SWAP` gates —
an NP-hard optimization. `quroute` learns the connectivity constraints of a target topology
and decides (a) an initial qubit mapping and (b) a SWAP-insertion strategy that minimizes
final **circuit depth** and **CNOT count**.

It drops into a Qiskit `PassManager` like any built-in pass and is benchmarked against
Qiskit's own `SabreSwap` / `SabreLayout`.

## Status

| Stage | What | State |
|------|------|-------|
| A | RL routing environment + simple-feature policy | **done** — env + REINFORCE policy (3x improvement over random) |
| B | GNN topology encoder inside the policy | **done** — size-agnostic GNN, zero-shot topology transfer |
| Plan B | imitation learning (imitate SABRE) if PPO is unstable | documented fallback |

## Install

```bash
git clone https://github.com/Canming921/quroute
cd quroute
pip install -e .            # baseline only (qiskit + numpy)
pip install -e ".[learn]"   # + torch / torch-geometric / gymnasium for Stage B
pip install -e ".[dev]"     # + pytest / ruff for development
```

## Quickstart

```python
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from quroute import QurouteRouter, GreedyShortestPathRouter, grid_topology

qc = QuantumCircuit(6)
qc.cx(0, 5)          # not adjacent on a 2x3 grid
qc.cx(1, 4)

cm = grid_topology(2, 3)
pm = PassManager([QurouteRouter(cm, GreedyShortestPathRouter())])
routed = pm.run(qc)  # every 2-qubit gate now lives on a coupling edge

print("inserted SWAPs:", pm.property_set["quroute_n_swaps"])
```

See [`examples/quickstart.py`](examples/quickstart.py) for a side-by-side SABRE comparison.

## Reinforcement-learning routing (Stage A)

```python
from quroute import RoutingEnv, GreedyDistancePolicy, PolicyRouter, grid_topology
from quroute.agents.policy_router import run_episode

cm = grid_topology(3, 3)
env = RoutingEnv(cm, circuit=my_circuit)          # gym-style MDP
policy = GreedyDistancePolicy(env.n_phys, env.num_actions)
result = run_episode(env, policy)                 # -> RoutedResult
# a trained neural policy plugs in identically:  PolicyRouter(trained_policy)
```

Train the learned policy (needs `[learn]`):  `python examples/train_reinforce.py`

## Learned GNN router (Stage B)

The Stage-A MLP is tied to one device. Stage B encodes the **coupling graph** with a GNN,
so the *same* trained policy routes on topologies of any size or shape — train on small
devices, evaluate on larger unseen ones with no retraining.

```python
from quroute import LearnedRouter, grid_topology, ring_topology
from quroute.train.gnn import train_gnn, GNNMaskedPolicy

net = train_gnn([grid_topology(3, 3), ring_topology(6)], iterations=1200)   # mixed topologies
router = LearnedRouter(GNNMaskedPolicy(net))      # drop-in BaseRouter, usable in QurouteRouter
```

Full demo (train + benchmark vs SABRE + zero-shot transfer table):
`python examples/train_and_benchmark_gnn.py`

Honest status: the learned policy matches the greedy heuristic on grids and generalizes
zero-shot to unseen topologies (beating random by 2-4x), but does not yet beat SABRE; the
remaining gap and concrete next steps (PPO, look-ahead reward, GAT) are in `docs/physics/`.

## Benchmarks vs SABRE

```bash
python examples/run_benchmark.py     # prints a table, writes CSV + figures
```

Standard suite (QFT / GHZ / QAOA-MaxCut / random), all pre-decomposed to <=2q gates with
no SWAPs so routing overhead is clean. SABRE gets best-of-5 seeds and a trivial layout, to
match our routers' trivial layout (a routing-only comparison). Added CNOTs, 3x3 grid:

| circuit | topological greedy | front-layer greedy | SABRE (best5) |
|---------|-------------------:|-------------------:|--------------:|
| qft     | 177 | 89 | 72 |
| qaoa    |  48 | 29 | 21 |
| random  |  49 | 37 | 28 |

SABRE still leads; closing that gap is the explicit goal of the Stage-B GNN policy.

## Architecture

```
QuantumCircuit ──▶ QurouteRouter (TransformationPass)
                        │  uses a BaseRouter:
                        ├─ GreedyShortestPathRouter   (Stage A baseline, working)
                        └─ LearnedRouter (RL + GNN)   (Stage B, planned)
                                 ▲
        topology.py (CouplingMap) + dag.py (front layer / interaction graph)
```

## Physics & algorithm notes

The "why" — SWAP as state exchange, its 3-CNOT decomposition in Dirac notation, why
minimizing depth fights decoherence on NISQ hardware — lives in
[`docs/physics/`](docs/physics/) (and will mirror to the GitHub Wiki).

## How AI was used

This project is built with an LLM as a *collaborating researcher*, not a ghost-writer.
Every non-trivial prompt, dead end, and design decision is logged in
[`AI-Collaboration.md`](AI-Collaboration.md).

## License

MIT — see [LICENSE](LICENSE).
