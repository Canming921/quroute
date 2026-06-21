"""Stage A / B 的训练代码。需要 `[learn]` 可选依赖(torch)。

特意不放进顶层 `quroute` 的导入路径,使基础包仅靠 qiskit + numpy 即可安装运行。
请显式导入:  `from quroute.train.reinforce import ...`
"""
