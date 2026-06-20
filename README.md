# quroute

**面向受限拓扑量子芯片的 AI 驱动比特映射与路由 —— 一个 Qiskit 编译器插件。**

量子算法默认所有比特两两相连,但真实芯片做不到。要让一个两比特门在物理上可执行,必须用 SWAP 门把比特状态搬到相邻位置——这是一个 NP-hard 优化问题。`quroute` 学习目标拓扑的连通性约束,自动决定 (a) 初始比特映射和 (b) SWAP 插入策略,以最小化最终的电路深度和 CNOT 门数量。

它像任何内置 pass 一样接入 Qiskit 的 `PassManager`,并与 Qiskit 自带的 `SabreSwap` / `SabreLayout` 做性能对比。

## 当前进度

| 阶段 | 内容 | 状态 |
|------|------|------|
| A | 强化学习路由环境 + 简单特征策略 | **完成** —— 环境 + REINFORCE 策略(比随机好约 3 倍)|
| B | 在策略中用 GNN 编码拓扑 | **完成** —— 尺寸无关的 GNN,可零样本跨拓扑迁移 |
| Plan B | 模仿学习(模仿 SABRE),训练不稳时的兜底方案 | 已在文档中记录 |

## 安装

```bash
git clone https://github.com/Canming921/quroute
cd quroute
pip install -e .            # 仅基础功能(qiskit + numpy)
pip install -e ".[learn]"   # + torch,用于 Stage B 的 GNN
pip install -e ".[dev]"     # + pytest / ruff,用于开发与测试
```

## 快速上手

```python
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from quroute import QurouteRouter, GreedyShortestPathRouter, grid_topology

qc = QuantumCircuit(6)
qc.cx(0, 5)          # 在 2x3 网格上不相邻
qc.cx(1, 4)

cm = grid_topology(2, 3)
pm = PassManager([QurouteRouter(cm, GreedyShortestPathRouter())])
routed = pm.run(qc)  # 现在每个两比特门都落在耦合边上

print("插入的 SWAP 数:", pm.property_set["quroute_n_swaps"])
```

## 强化学习路由(Stage A)

把 front layer 包装成一个 gym 式 MDP:状态 = 当前映射 + 待执行门 + 拓扑;动作 = 选一条边做 SWAP;奖励 = SWAP 成本 + 势函数式距离整形(等价于"最小化 SWAP 总数")。

```python
from quroute import RoutingEnv, GreedyDistancePolicy, PolicyRouter, grid_topology
from quroute.agents.policy_router import run_episode

cm = grid_topology(3, 3)
env = RoutingEnv(cm, circuit=my_circuit)
policy = GreedyDistancePolicy(env.n_phys, env.num_actions)
result = run_episode(env, policy)        # -> RoutedResult
```

## 学习型 GNN 路由器(Stage B)

Stage A 的 MLP 与单一设备绑死。Stage B 用 GNN 直接编码**耦合图**,因此同一个训练好的策略能在任意大小、任意形状的拓扑上路由——在小设备上训练,可直接迁移到没见过的大设备,无需重训。

```python
from quroute import LearnedRouter, grid_topology, ring_topology
from quroute.train.gnn import train_gnn, GNNMaskedPolicy

net = train_gnn([grid_topology(3, 3), ring_topology(6)], iterations=1200)  # 混合拓扑训练
router = LearnedRouter(GNNMaskedPolicy(net))   # 可直接放进 QurouteRouter 使用
```

完整示例(训练 + 对比 SABRE + 零样本迁移表):`python examples/train_and_benchmark_gnn.py`

诚实说明:学习到的策略在网格拓扑上能追平贪心启发式,并能零样本泛化到没训练过的拓扑(比随机好 2–4 倍),但**尚未超过 SABRE**;剩余差距与后续改进路线(PPO、look-ahead 奖励、GAT 等)写在 `docs/physics/` 里。

## 对比 SABRE 的 Benchmark

```bash
python examples/run_benchmark.py     # 打印对比表,生成 CSV 和图
```

标准电路集(QFT / GHZ / QAOA-MaxCut / 随机),全部预先分解为 ≤2 比特门且不含 SWAP,使路由开销可被干净地归因。SABRE 取 5 个 seed 的最优,且与本项目一样使用 trivial 初始映射(纯路由对比)。新增 CNOT 数(3×3 网格):

| 电路 | 拓扑贪心 | front 贪心 | SABRE(best5) |
|------|--------:|----------:|-------------:|
| qft  | 177 | 89 | 72 |
| qaoa |  48 | 29 | 21 |
| random | 49 | 37 | 28 |

SABRE 仍领先;缩小这道差距正是 Stage B GNN 策略的明确目标。

## 架构

```
QuantumCircuit ──▶ QurouteRouter(TransformationPass)
                        │  内部使用一个 BaseRouter:
                        ├─ GreedyShortestPathRouter   (Stage A 基线,可用)
                        └─ LearnedRouter(RL + GNN)    (Stage B)
                                 ▲
        topology.py(耦合图) + dag.py(front layer / 交互图)
```

## 物理与算法说明

核心物理推导(SWAP 作为态交换、它的 3-CNOT 分解、为什么压低深度能对抗 NISQ 退相干)放在 [`docs/physics/`](docs/physics/),并会同步到 GitHub Wiki。

## AI 是如何参与的

本项目把大模型当作"协作研究员"而非代写工具。每一个关键 prompt、走过的弯路、设计决策都记录在 [`AI-Collaboration.md`](AI-Collaboration.md)。

## 许可证

MIT —— 见 [LICENSE](LICENSE)。
