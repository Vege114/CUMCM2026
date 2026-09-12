# exp006 预备实验：决策树误差分布与离散 SOC 动态规划

当前状态：**仅完成小规模测速，等待用户确认后才开展全年实验**。没有正式年度成绩，没有重训 exp004 预测，也没有登记到历史排行榜。

独立分支 `codex/q2-tree-planning` 从预测分支 `codex/q2-discussion-seasonality` 的 `b42168d5271097762953f38d472d0ef5fe1a908d` 建立。独立 worktree 位于项目 `.work/q2-tree-planning`；原目录继续留在 `codex/q2-linear-planning`。本实验不得写原目录中的代码、结果、报告或运行日志。

## 已有实现

- `risk.py`：冻结 exp004 预测；只使用午夜前完整揭晓的最近 28 个历史预报日，训练深度 5、叶节点至少 48 样本的回归树，生成每槽 9 点等权净负荷边缘分布。首两天缺少同源预报误差时，明确使用历史周期基线误差回退。
- `model.py`：有符号电量变化保证互斥；离散 SOC 动态规划最小化计划费与期望五倍紧急费，另加轻微吞吐和方向反转罚。午夜锁定全部购电量，执行时根据当前实际供需和实际 SOC 裁剪动作，绝不为充电额外购买紧急电。
- `benchmark.py`：8 个固定日期、100/50/25 kWh 网格、3 次重复，外加两日各三个离网初值检查；只有测速入口，**没有全年运行选项**。
- `tests/test_q2_tree_planning.py`：购电分位数对比全部折点、DP 对比两步穷举、极端实况物理约束与执行前缀因果性、未来实况变更不影响午夜误差分布。

单槽分位数公式仅对确定储能动作下的离散开环替代模型成立。实际动作会投影，故不能将 DP 目标宣称为全年真实费用全局最优。树是误差分布估计器，DP 是规划器，此路线不属于强化学习。九点压缩分布的 0.8 分位对应第八点，约为原叶节点残差的 0.8333 分位，应披露此近似。

## 测速复现

在这个独立 worktree 根目录执行（复用原项目 Python 环境，只读依赖）：

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 \
  /Users/vegetarianwolf/Projects/CUMCM2026/.venv/bin/python \
  -m unittest discover -s tests -p 'test_q2_tree_planning.py' -v

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 \
  /Users/vegetarianwolf/Projects/CUMCM2026/.venv/bin/python \
  -m experiments.problem2.tree_planning.benchmark --repeats 3
```

计算使用单 CPU 线程，不启动 TensorFlow/GPU、不调用 MILP/LP。当前沙箱不允许调整进程 nice 值，故实际未应用优先级修改；并行运行时仍可能争用 CPU，不能承诺零影响。

[测速及正式工作预算](../../../reports/experiments/exp006/preflight.md) · [机器可读测速](../../../data/results/exp006/pilot/benchmark.json) · [待批准的正式配置](protocol.proposed.json)

报告和正式运行入口将在用户批准后完成；不以测试日期的费用挑选最优参数。该工作树包含完整历史报告证据，可从现有八节模板制作正式报告而不改写 exp001–005。
