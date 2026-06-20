# AI Collaboration Log

> Required by the course rubric (10%). This file documents how we used a large language
> model as a **collaborating researcher** — including prompts, dead ends, corrections,
> and design decisions — rather than as a blind code generator.

Format per entry: date · goal · what we asked / iterated · what we kept vs rejected · why.

---

## 2026-06-18 · Project framing & method choice
- **Goal:** decide RL vs GNN vs hybrid for the routing agent.
- **Iteration:** asked the model to map the rubric weights to a strategy; pushed back on
  "just beat SABRE" framing. Model argued performance is *not* the dominant rubric axis
  (30% physics/architecture understanding, 25% engineering, 20% commit history).
- **Decision:** RL (PPO) as the decision backbone + GNN as the topology-state encoder,
  delivered in two stages (simple-feature MLP first, GNN second). Rationale: routing is a
  sequential decision problem (needs RL), and the rubric explicitly rewards "learning the
  constraint features of the topology" (needs a GNN over the coupling graph).
- **Rejected:** pure GNN (not a sequential decision-maker; caps at imitation of SABRE).
- **Kept as Plan B:** imitation learning if PPO training is unstable near the deadline.

## 2026-06-18 · Scaffold + CI
- **Goal:** stand up a real, testable Qiskit plugin skeleton (not a placeholder).
- **Iteration:** verified the live Qiskit 2.4 API in a sandbox (`CouplingMap.distance`,
  `shortest_undirected_path`, `dag.front_layer`, `dag.two_qubit_ops`) before writing code,
  rather than trusting the model's memory of an older API.
- **Decision:** ship a working `GreedyShortestPathRouter` baseline now so the pipeline runs
  end-to-end from commit #1; keep `LearnedRouter` as a documented stub.

---

### TODO log template (fill as you go)
- [ ] Stage A RL env: state/action/reward design — prompts & results
- [ ] Reward shaping experiments — what worked / what diverged
- [ ] GNN encoder choice (GAT vs GINEConv) — comparison
- [ ] SABRE benchmark methodology — fairness checks the model flagged

## 2026-06-18 (cont.) · Stage A RL environment
- **Goal:** turn the front layer into a gym-style MDP and get a learning signal.
- **Iteration:**
  - Designed reward as swap-cost + potential-based distance shaping so that maximizing
    return provably equals minimizing SWAPs (checked the shaping theorem, not just vibes).
  - First training run: deterministic greedy *evaluation* livelocked in a 2-cycle
    (swap an edge, then immediately undo it). Diagnosed and fixed by forbidding the
    immediately-previous action at inference + random tie-breaking in the scripted policy.
  - Verified the env runs with numpy+qiskit only (kept torch out of the import path) so CI
    stays light; torch tests auto-skip when torch is absent.
- **Result:** trained MLP policy ~8.3 mean swaps vs ~19 random on a 3x3 grid (30 circuits),
  and front-layer greedy (9 swaps) already beats topological greedy (10) on the demo,
  closing on SABRE (8). Honest baseline curve to report.
- **Decision:** keep MLP as Stage-A baseline; Stage B swaps only the *encoder* for a GNN.

## 2026-06-18 (cont.) · Benchmark framework
- **Goal:** a fair, reproducible comparison vs Qiskit SABRE with real numbers + figures.
- **Iteration:**
  - Caught that `qiskit.circuit.library.QFT` is deprecated (removal in Qiskit 3.0);
    switched to `qiskit.synthesis.synth_qft_full`.
  - Fairness decisions made explicit: (1) pre-decompose all benchmark circuits to <=2q
    with NO swap gates, so any output SWAP is pure routing overhead; (2) compute metrics by
    flattening SWAP->CX and measuring on one common basis; (3) give SABRE best-of-5 seeds
    and a trivial layout, matching our routers' trivial layout (routing-only comparison).
  - Kept the runner numpy+qiskit only; plotting (matplotlib) is lazily imported so CI core
    stays light and the plot test auto-skips when matplotlib is absent.
- **Result:** SABRE leads on all families; front-layer greedy roughly halves the
  topological-greedy overhead on QFT (177->89 added CX at n=9). The SABRE gap is now
  quantified and becomes the explicit success criterion for Stage B.

## 2026-06-18 (cont.) · Stage B GNN policy
- **Goal:** replace the MLP encoder with a GNN so the policy learns topology constraints
  and transfers across devices.
- **Iteration / dead ends:**
  - First GNN beat random but lagged the greedy heuristic; root cause: it lacked the
    per-edge distance-reduction signal the heuristic uses. Added it as an *edge feature*
    (physically: "how much does this SWAP help the front layer now") -> GNN then matched
    greedy on grids.
  - Deterministic greedy inference livelocked on a line (unseen, low connectivity). A
    last-action guard wasn't enough on large graphs (too many candidate edges). Replaced
    with a *stall detector*: if no gate executes for `2N` steps, force the max-distance-
    reduction SWAP -> strictly decreases front-layer distance -> provable termination.
  - Verified the headline claim empirically: a net trained only on small topologies routes
    validly on 4x4 / ring-8 / line-8 it never saw (the MLP literally cannot, by input shape).
- **Decision / honesty:** report that the GNN matches the heuristic + generalizes, but does
  NOT beat SABRE yet; list concrete next steps rather than over-claim. The rubric rewards
  understanding + honest engineering over a leaderboard win.
