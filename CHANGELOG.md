# Changelog

## [0.4.0] - Stage B: GNN policy
### Added
- `quroute.train.gnn`: a from-scratch GCN policy (no torch-geometric) over the device
  coupling graph + an edge scorer that also reads the per-edge front-layer distance
  reduction. Size-agnostic: one trained net runs on any topology.
- `RoutingEnv` now exposes graph state in `info` (`node_features`, `adjacency`,
  `action_edges`) plus `node_features()` and cached normalized adjacency / degree.
- `GNNMaskedPolicy` with a stall-detector + tabu giving a provable termination guarantee
  on any connected topology (incl. low-connectivity lines never trained on).
- `LearnedRouter` is now implemented: wraps a trained policy as a drop-in `BaseRouter`.
- REINFORCE trainer with entropy bonus; trains across a list of topologies.
- `examples/train_and_benchmark_gnn.py`; GNN tests (skip without torch); version 0.4.0.
- `[learn]` extra slimmed to just `torch` (GNN implemented from scratch).

### Result (honest)
- GNN matches the greedy heuristic on grids (~7.1 vs 7.0 added-CX-equiv on 3x3) and
  generalizes zero-shot to unseen 4x4 / ring-8 / line-8 (2-4x better than random).
- It does NOT yet beat SABRE or the heuristic on lines. Next steps (PPO/GAE, look-ahead
  reward, GAT/GINEConv, learned layout) documented in docs/physics.

## [0.3.0] - Benchmark framework
### Added
- `quroute.benchmark`: standard circuit generators (QFT via non-deprecated
  `synth_qft_full`, GHZ, QAOA-MaxCut, random), all pre-decomposed to <=2q gates with no
  SWAPs so routing overhead is cleanly attributable.
- `benchmark()` runner + `circuit_metrics` (flatten SWAP->CX, then compare added CX /
  2q-depth on a common basis) + `aggregate` / `summarize` / `to_csv`.
- `SabreBaselineRouter` (best-of-N-seed SABRE) wrapped as a `BaseRouter` for a fair,
  uniform, routing-only comparison from a trivial layout.
- `benchmark/plots.py` (matplotlib, lazy import): overhead-by-circuit bars + scaling line.
- `examples/run_benchmark.py`; tests (plot test auto-skips without matplotlib).
- Version aligned to 0.3.0 across pyproject / __init__ / changelog.

### Measured (3x3 grid, mean of 3 instances, added CNOTs; lower is better)
| circuit | topological greedy | front-layer greedy | SABRE (best5) |
|---------|-------------------:|-------------------:|--------------:|
| qft     | 177 | 89 | 72 |
| qaoa    |  48 | 29 | 21 |
| random  |  49 | 37 | 28 |
| ghz     |  21 | 21 | 12 |

SABRE still leads — the visible gap is the target for the Stage-B GNN policy.

## [0.2.0] - Stage A: RL environment
### Added
- `RoutingEnv`: gym-style routing MDP (numpy+qiskit only, CI-safe). State = engineered
  features (occupancy, front-layer involvement, per-edge distance reduction, progress);
  action = device edge to SWAP; reward = swap cost + potential-based distance shaping.
- `policies.py`: `RandomMaskedPolicy`, `GreedyDistancePolicy` (front-layer-aware, SABRE-lite).
- `agents.PolicyRouter` + `run_episode`: any policy becomes a drop-in `BaseRouter`.
- `quroute.train.reinforce`: masked-MLP policy + REINFORCE trainer (needs `[learn]`/torch),
  with anti-livelock guard. Measured: trained ~8.3 vs random ~19 mean swaps (3x3 grid).
- Tests for the env + a torch smoke test (auto-skipped without torch).
- `examples/train_reinforce.py`; quickstart now compares 3 routers vs SABRE.

## [0.1.0] - scaffold
### Added
- Project skeleton, packaging, CI/CD; topology + DAG utilities; `BaseRouter`/`RoutedResult`;
  `GreedyShortestPathRouter` baseline; `QurouteRouter` Qiskit pass; `LearnedRouter` stub.

### Next
- Stage B: swap the MLP encoder for a GNN over (coupling graph, interaction graph).
- Benchmark harness: standardized circuit suite (QAOA / QFT / random) vs SABRE.
