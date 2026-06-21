# 更新日志

## [0.4.0] - Stage B:GNN 策略
### 新增
- `quroute.train.gnn`:从零实现的 GCN 策略(不依赖 torch-geometric),作用在设备耦合图上,
  其边打分器还读取每条边的 front-layer 距离削减量。尺寸无关:一个训练好的网络可在任意拓扑上运行。
- `RoutingEnv` 现在在 `info` 中暴露图状态(`node_features`、`adjacency`、`action_edges`),
  并新增 `node_features()` 以及缓存的归一化邻接 / 度数。
- `GNNMaskedPolicy` 带停滞检测 + tabu,在任意连通拓扑上(含从未训练过的低连通度线形)可证收敛。
- `LearnedRouter` 已实现:把训练好的策略包装成即插即用的 `BaseRouter`。
- 带熵正则的 REINFORCE 训练器;可跨多个拓扑训练。
- `examples/train_and_benchmark_gnn.py`;GNN 测试(无 torch 时跳过);版本升到 0.4.0。
- `[learn]` 可选依赖精简为仅 `torch`(GNN 从零实现)。

### 结果(诚实表述)
- GNN 在网格上追平贪心启发式(3x3 上约 7.1 vs 7.0 的等效新增 CX),并能零样本泛化到
  未见过的 4x4 / ring-8 / line-8(比随机好 2~4 倍)。
- 它尚未超过 SABRE,也未在线形拓扑上超过启发式。后续路线(PPO/GAE、look-ahead 奖励、
  GAT/GINEConv、可学习初始布局)记录在 docs/physics 中。

## [0.3.0] - Benchmark 框架
### 新增
- `quroute.benchmark`:标准电路生成器(QFT 用未弃用的 `synth_qft_full`、GHZ、QAOA-MaxCut、随机),
  全部预先分解为 ≤2 比特门且不含 SWAP,使路由开销可被干净归因。
- `benchmark()` 运行器 + `circuit_metrics`(把 SWAP 展开成 CX,再在统一基门上比较新增 CX / 两比特门深度)
  + `aggregate` / `summarize` / `to_csv`。
- `SabreBaselineRouter`(取多 seed 最优的 SABRE)包装成 `BaseRouter`,做公平、统一、仅路由的对比。
- `benchmark/plots.py`(matplotlib,懒加载):按电路分组的柱状图 + 随规模变化的折线图。
- `examples/run_benchmark.py`;测试(无 matplotlib 时跳过绘图测试)。
- 版本在 pyproject / __init__ / changelog 间统一为 0.3.0。

### 实测(3x3 网格,3 个实例均值,新增 CNOT,越低越好)
| 电路 | 拓扑贪心 | front-layer 贪心 | SABRE(best5) |
|------|--------:|-----------------:|-------------:|
| qft  | 177 | 89 | 72 |
| qaoa |  48 | 29 | 21 |
| random | 49 | 37 | 28 |
| ghz  |  21 | 21 | 12 |

SABRE 仍领先 —— 那道可见的差距正是 Stage B GNN 策略的目标。

## [0.2.0] - Stage A:强化学习环境
### 新增
- `RoutingEnv`:gym 式路由 MDP(仅依赖 numpy+qiskit,CI 安全)。状态 = 工程化特征
  (占用、是否在 front layer、每条边的距离削减、进度);动作 = 要 SWAP 的设备边;
  奖励 = SWAP 成本 + 势函数式距离整形。
- `policies.py`:`RandomMaskedPolicy`、`GreedyDistancePolicy`(感知 front layer,即 SABRE-lite)。
- `agents.PolicyRouter` + `run_episode`:任意策略都能变成即插即用的 `BaseRouter`。
- `quroute.train.reinforce`:带掩码的 MLP 策略 + REINFORCE 训练器(需 `[learn]`/torch),带防死循环守卫。
  实测:训练后约 8.3 vs 随机约 19 的平均 SWAP 数(3x3 网格)。
- 环境测试 + 一个 torch 冒烟测试(无 torch 时自动跳过)。
- `examples/train_reinforce.py`;quickstart 现在对比 3 种路由器 vs SABRE。

## [0.1.0] - 项目骨架
### 新增
- 项目骨架、打包、CI/CD;拓扑 + DAG 工具;`BaseRouter`/`RoutedResult`;
  `GreedyShortestPathRouter` 基线;`QurouteRouter` Qiskit pass;`LearnedRouter` 占位。
