# Physics & algorithm notes

These notes back the 30% "physical mechanism + algorithm architecture" rubric item.
They will be mirrored to the GitHub Wiki. Keep explanations in precise QM language
(Dirac notation) and tie every code module to the physical picture behind it.

## 1. Why routing exists
A two-qubit gate (e.g. CNOT) can only be applied to **physically coupled** qubits. The
device connectivity is a graph $G_\text{phys}=(V,E)$. If a logical gate needs two qubits
that are not adjacent, their states must first be moved together.

## 2. The SWAP gate (to be written)
- $\mathrm{SWAP}\,|\psi\rangle|\varphi\rangle = |\varphi\rangle|\psi\rangle$.
- Matrix form, and the standard decomposition into 3 CNOTs.
- Cost model: 1 SWAP = 3 CNOTs on a CX-native basis (this is `RoutedResult.added_cx`).

## 3. Why minimize depth & CNOT count (to be written)
- NISQ decoherence ($T_1$, $T_2$); per-CNOT error (~$10^{-2}$–$10^{-3}$); error accumulation.
- Depth vs. total-gate-count trade-offs (SABRE minimizes SWAPs; depth can differ).

## 4. The mapping/routing problem as search (to be written)
- Initial layout, front layer, look-ahead.
- SABRE heuristic recap; where our RL+GNN agent differs.

## 5. Module ↔ physics map (to be filled)
| code module | physical concept |
|-------------|------------------|
| `topology.py` | connectivity graph $G_\text{phys}$ |
| `dag.py` (front layer / interaction graph) | executable gates & $G_\text{int}$ |
| `agents/greedy.py` | shortest-path SWAP insertion |
| `agents/learned.py` | learned policy over $(G_\text{phys}, G_\text{int})$ |

## 6. Routing as a Markov Decision Process (Stage A)
We cast SWAP insertion as an MDP solved by reinforcement learning:
- **State** $s_t$: the current mapping $\pi_t:\text{logical}\to\text{physical}$, the front
  layer $F_t$ (gates with all predecessors executed), and the device graph $G_\text{phys}$.
- **Action** $a_t$: choose an edge $(i,j)\in E$ and apply $\mathrm{SWAP}_{ij}$, updating
  $\pi_{t+1}$. The candidate set is restricted to edges incident to a front-layer qubit.
- **Reward** $r_t = c_\text{swap} + \alpha\,\big(D(s_t)-D(s_{t+1})\big)$, where
  $D(s)=\sum_{(u,v)\in F}\mathrm{dist}_{G_\text{phys}}(\pi(u),\pi(v))$ is the total
  front-layer distance. With $c_\text{swap}=-1$ and the distance term as
  *potential-based shaping*, maximizing return is equivalent to minimizing total SWAPs
  while keeping the optimal policy unchanged (Ng et al., shaping theorem).
- **Episode** ends when $F=\varnothing$ (all gates routed). The greedy-on-$D$ policy is a
  SABRE-lite baseline; the learned policy improves on it; Stage B replaces the hand-built
  per-edge feature $D(s_t)-D(s_{t+1})$ with a GNN over $(G_\text{phys}, G_\text{int})$.

## 7. The GNN policy (Stage B)
The Stage-A MLP reads a flat feature vector whose width is tied to one device, so it
cannot transfer. Stage B replaces the encoder with a graph neural network that operates
directly on the device coupling graph $G_\text{phys}$, which is exactly "learning the
constraint features of the target topology".

**Message passing.** With node features $X\in\mathbb{R}^{N\times 4}$ (occupied,
in-front-layer, normalized distance-to-partner, normalized degree) and the
symmetric-normalized adjacency $\hat A = \tilde D^{-1/2}(A+I)\tilde D^{-1/2}$, each GCN
layer computes $H^{(\ell+1)}=\sigma\!\big(\hat A\,H^{(\ell)}W^{(\ell)}\big)$. After $L$
layers, node $v$'s embedding summarizes its $L$-hop topological neighbourhood.

**Edge (action) scoring.** For a candidate SWAP on edge $(a,b)$ the logit is
$\,\phi\big([\,h_a+h_b\,;\,\Delta_{ab}\,]\big)$, symmetric in $a,b$, where $\Delta_{ab}$
is the front-layer distance reduction that swap would produce — the same physical signal
the greedy heuristic uses. Masking restricts to candidate edges; a categorical over the
masked logits gives the policy.

**Why it transfers.** All weights ($W^{(\ell)}$, $\phi$) act per-node / per-edge and are
shared, so the *same* trained network runs on a graph of any size or shape. We train on a
mix of small topologies and evaluate on larger, unseen ones (4x4 grid, ring-8, line-8)
with no retraining — impossible for the fixed-width MLP.

**Honest status.** With REINFORCE + a small GCN, the learned policy matches the greedy
heuristic on grids and generalizes zero-shot, but does not yet beat SABRE or the heuristic
on low-connectivity lines. Documented next steps: PPO/GAE (lower-variance updates), a
decay-style look-ahead term in the reward (as in SABRE), GAT/GINEConv layers, a learned
initial layout, and curriculum over topologies.
