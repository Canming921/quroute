# 物理机制与算法解析

> 全文用精确的量子力学语言(狄拉克符号)阐述,并把每个核心代码模块与其
> 背后的物理图景一一对应。

## 1. 为什么需要路由(SWAP 的物理起源)

一个两比特门(如 CNOT)在物理上只能作用于**彼此耦合(相邻)**的两个比特。设备的连通性由一张图描述:

$$G_\text{phys} = (V, E),$$

其中 $V$ 是物理比特, $E$ 是允许直接做两比特门的比特对。量子算法在设计时通常假设全连接,但真实芯片的 $E$ 非常稀疏(线形、网格、heavy-hex 等)。

当一个逻辑门需要的两个比特在 $G_\text{phys}$ 上不相邻时,必须先把它们的量子态搬到相邻位置再执行门。搬运量子态的工具就是 **SWAP 门**。如何选择初始映射、以及在何处插入哪些 SWAP,使最终电路的深度和 CNOT 总数最小,是一个 NP-hard 优化问题——这正是本项目要解决的核心。

## 2. SWAP 门:态交换与 3-CNOT 分解

SWAP 门交换两个比特的量子态。用狄拉克符号表示:

$$\mathrm{SWAP}\,|\psi\rangle_a |\varphi\rangle_b = |\varphi\rangle_a |\psi\rangle_b .$$

在计算基 $\{|00\rangle, |01\rangle, |10\rangle, |11\rangle\}$ 下,其矩阵为:

$$
\mathrm{SWAP} =
\begin{pmatrix}
1 & 0 & 0 & 0\\
0 & 0 & 1 & 0\\
0 & 1 & 0 & 0\\
0 & 0 & 0 & 1
\end{pmatrix}.
$$

它只交换 $|01\rangle \leftrightarrow |10\rangle$,而让 $|00\rangle$、$|11\rangle$ 不变,正是"交换两比特态"的体现。

硬件通常没有原生 SWAP,需用 3 个 CNOT 实现:

$$\mathrm{SWAP}_{ab} = \mathrm{CNOT}_{a\to b}\,\mathrm{CNOT}_{b\to a}\,\mathrm{CNOT}_{a\to b}.$$

逐步验证(CNOT 表示:控制比特为 1 时翻转目标比特):

$$|10\rangle \xrightarrow{\mathrm{CNOT}_{a\to b}} |11\rangle \xrightarrow{\mathrm{CNOT}_{b\to a}} |01\rangle \xrightarrow{\mathrm{CNOT}_{a\to b}} |01\rangle$$

即 $|10\rangle \to |01\rangle$;

$$|01\rangle \xrightarrow{\mathrm{CNOT}_{a\to b}} |01\rangle \xrightarrow{\mathrm{CNOT}_{b\to a}} |11\rangle \xrightarrow{\mathrm{CNOT}_{a\to b}} |10\rangle$$

即 $|01\rangle \to |10\rangle$;而 $|00\rangle$ 与 $|11\rangle$ 保持不变。

这正是 SWAP 的作用。**代价模型**:在以 CX 为原生门的设备上,1 个 SWAP = 3 个 CNOT。代码中 `RoutedResult.added_cx` 即按 $3\times(\text{SWAP 数})$ 计算。

## 3. 为什么要最小化深度与 CNOT 数

在 NISQ(含噪中等规模量子)硬件上,比特会退相干:

- $T_1$(能量弛豫时间): $|1\rangle$ 自发衰减到 $|0\rangle$;
- $T_2$(相位相干时间):叠加态的相对相位被随机化(失相)。

电路**深度**越大,执行时间越长,比特暴露在退相干下的时间越长,保真度越低。同时每个 CNOT 都带有约 $10^{-2}\sim10^{-3}$ 的门误差,误差随门数累积。由于 1 个 SWAP = 3 个 CNOT,路由阶段插入的每个 SWAP 都会显著放大误差。

因此优化目标是同时压低**最终电路深度**和 **CNOT 总数**。注意两者并不完全等价:SABRE 主要最小化 SWAP 数,但深度还取决于这些 SWAP 能否并行,所以本项目在 benchmark 中同时报告两个指标。

## 4. 映射与路由问题:图上的搜索

整个问题包含两个相互耦合的子问题:

1. **初始布局(initial layout)**:把逻辑比特映射到物理比特 $\pi_0:\text{logical}\to\text{physical}$。好的初始布局能让更多门一开始就相邻。(本项目目前用 trivial 布局,可学习布局列为后续工作。)
2. **路由(routing)**:在执行过程中动态插入 SWAP。

关键概念 **front layer(前沿层)**:DAG(有向无环图)中所有前驱都已执行的门集合,即"当前可执行的门"。路由器反复执行:先做掉 front layer 中已相邻的门 → 对仍不相邻的门选择 SWAP 把它们拉近 → 更新 front layer,直到所有门执行完毕。

**SABRE** 是业界标准基线:它通过正反向遍历得到较好的初始布局,并用带 look-ahead(展望后续门)和衰减项的启发式打分来选择 SWAP。本项目用 RL/GNN 学习"选哪个 SWAP"这一决策,与 SABRE 的对比见 benchmark。

## 5. 代码模块 ↔ 物理概念对照

| 代码模块 | 对应的物理 / 算法概念 |
|----------|----------------------|
| `topology.py` | 设备连通图 $G_\text{phys}$、距离矩阵 |
| `dag.py` | front layer(可执行门)、交互图 $G_\text{int}$ |
| `agents/greedy.py` | 沿最短路径插入 SWAP 的朴素基线 |
| `env.py` | 把路由建模为 MDP(见第 6 节) |
| `agents/learned.py` + `train/gnn.py` | 在 $(G_\text{phys}, G_\text{int})$ 上的学习型策略(见第 7 节) |

## 6. 把路由建模为马尔可夫决策过程(Stage A)

我们把 SWAP 插入建模为一个用强化学习求解的 MDP:

- **状态** $s_t$:当前映射 $\pi_t$、front layer $F_t$、以及设备图 $G_\text{phys}$。
- **动作** $a_t$:从与 front layer 比特相邻的边中选一条 $(i,j)\in E$,执行该边上的 SWAP 并更新映射;候选集被限制在这些边上,大幅缩小动作空间。
- **奖励** $r_t = c_\text{swap} + \alpha\big(D(s_t)-D(s_{t+1})\big)$,其中

$$D(s) = \sum_{(u,v)\in F} \mathrm{dist}_{G_\text{phys}}\big(\pi(u),\pi(v)\big)$$

是 front layer 中各门两端在设备图上的总距离。取 $c_\text{swap}=-1$,并把距离项作为**势函数式整形(potential-based shaping)**;由 Ng 等人的整形定理可知,这样不改变最优策略,且"最大化累计回报"严格等价于"最小化 SWAP 总数"。

- **回合终止**: $F=\varnothing$(所有门执行完)。

按 $D$ 贪心的策略就是一个 SABRE-lite 基线;学习型策略在此之上改进。

## 7. GNN 策略(Stage B)

Stage A 的 MLP 读取一个扁平特征向量,其维度与某一台具体设备绑定,因此无法迁移。Stage B 把编码器换成直接作用在耦合图 $G_\text{phys}$ 上的图神经网络——这正是评分表所说的"学习目标拓扑的约束特征"。

**消息传递**:节点特征 $X\in\mathbb{R}^{N\times 4}$(是否被占用、是否在 front layer、到配对比特的归一化距离、归一化度数);取带自环的对称归一化邻接 $\hat A = \tilde D^{-1/2}(A+I)\tilde D^{-1/2}$,每层 GCN 计算

$$H^{(\ell+1)} = \sigma\!\big(\hat A\, H^{(\ell)} W^{(\ell)}\big).$$

经过 $L$ 层后,节点 $v$ 的嵌入概括了它 $L$ 跳邻域内的拓扑信息。

**动作(边)打分**:候选 SWAP 边 $(a,b)$ 的 logit 为

$$\phi\big([\,h_a + h_b\,;\,\Delta_{ab}\,]\big),$$

对 $a,b$ 对称;其中 $\Delta_{ab}$ 是这次 SWAP 能带来的 front-layer 距离削减——也就是贪心启发式所用的同一个物理信号。掩码后在合法边上做 softmax 得到策略。

**为什么能迁移**:所有权重($W^{(\ell)}$、$\phi$)都是按节点 / 按边共享的,因此同一个训练好的网络能在任意大小、任意形状的图上运行。本项目在一组小拓扑上训练,直接在更大的、没见过的拓扑(4×4 网格、ring-8、line-8)上评估而无需重训——这是固定维度的 MLP 做不到的。

**诚实的现状**:用 REINFORCE + 小型 GCN,学习到的策略在网格上追平了贪心启发式并能零样本泛化,但尚未超过 SABRE,也未在低连通度的线形拓扑上超过启发式。已记录的后续改进路线:PPO/GAE(降低梯度方差)、SABRE 式 look-ahead 衰减奖励、GAT/GINEConv 层、可学习初始布局,以及拓扑课程学习。
