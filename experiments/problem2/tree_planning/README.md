# exp006：决策树误差分布与离散 SOC 动态规划

当前状态：**用户确认后，15 组正式、兼容与控制实验全部完成**。首次正式计算墙钟耗时 202.34 秒（3 分 22 秒）；冻结 exp004 预测，未重新训练预测模型。

预声明主组真实费用 14,066,257.48 元，比 exp004 无季节基线增加 454,669.64 元（3.34%）；方向反转减少 64.63%，EFC 仅减少 4.29%。本轮测得费用最低的是预声明的 `greedy_execution` 对照：13,534,196.73 元，节省 77,391.11 元（0.57%），方向反转减少 12.44%、EFC 减少约 1.96%。该对照只运行种子 42，不能宣称跨种子稳健或全局最优，也不替换预声明主组身份。全部方案实际同时充放电槽数为零；旧基线同样为零。

[完整报告](../../../reports/experiments/exp006/report.md) · [离线交互报告](../../../reports/experiments/exp006/report.html) · [费用优先对照工作簿](../../../reports/experiments/exp006/greedy_execution/result2.xlsx) · [主组工作簿](../../../reports/experiments/exp006/result2.xlsx) · [原始运行清单](../../../data/results/exp006/run_manifest.json)

独立分支 `codex/q2-tree-planning` 从预测分支 `codex/q2-discussion-seasonality` 的 `b42168d5271097762953f38d472d0ef5fe1a908d` 建立。独立 worktree 位于项目 `.work/q2-tree-planning`；原目录继续留在 `codex/q2-linear-planning`。本实验不得写原目录中的代码、结果、报告或运行日志。

## 已有实现

- `risk.py`：冻结 exp004 预测；只使用午夜前完整揭晓的最近 28 个历史预报日，训练深度 5、叶节点至少 48 样本的回归树，生成每槽 9 点等权净负荷边缘分布。首两天缺少同源预报误差时，明确使用历史周期基线误差回退。
- `model.py`：有符号电量变化保证互斥；离散 SOC 动态规划最小化计划费与期望五倍紧急费，另加轻微吞吐和方向反转罚。午夜锁定全部购电量，执行时根据当前实际供需和实际 SOC 裁剪动作，绝不为充电额外购买紧急电。
- `benchmark.py`：批准前的小规模测速入口；8 个固定日期、100/50/25 kWh 网格、3 次重复，外加两日各三个离网初值检查。
- `run.py`：按 `protocol.approved.json` 运行 15 个预声明组，连续传递 SOC、方向及功率状态，每 14 日保存带签名与哈希的检查点；`verify.py` 独立核验逐槽档案。
- `export.py`：按原有模版导出主组或费用优先对照工作簿和题目指定日表格，不重新求解。
- `tests/test_q2_tree_planning.py`：购电分位数对比全部折点、DP 对比两步穷举、极端实况物理约束与执行前缀因果性、未来实况变更不影响午夜误差分布。

单槽分位数公式仅对确定储能动作下的离散开环替代模型成立。实际动作会投影，故不能将 DP 目标宣称为全年真实费用全局最优。树是误差分布估计器，DP 是规划器，此路线不属于强化学习。九点压缩分布的 0.8 分位对应第八点，约为原叶节点残差的 0.8333 分位，应披露此近似。

## 检查归档与重建输出

在这个独立 worktree 根目录使用原项目环境：

```sh
/Users/vegetarianwolf/Projects/CUMCM2026/.venv/bin/python \
  -m experiments.problem2.tree_planning.verify data/results/exp006/primary

/Users/vegetarianwolf/Projects/CUMCM2026/.venv/bin/python \
  -m experiments.problem2.tree_planning.verify data/results/exp006/greedy_execution

/Users/vegetarianwolf/Projects/CUMCM2026/.venv/bin/python \
  -m experiments.problem2.tree_planning.export --case greedy_execution
```

两个测试文件合计 9 项通过；15 组的 721,440 个槽通过独立物理、费用及因果信息边界检查。工作簿另行验证公式缓存、变更输入重算、逐槽金额与实际渲染。重建报告使用 `reports/build_report_exp006.py`，不需要重跑规划。

`run.py` 固定写入本工作树的 `data/results/exp006`，支持签名一致的检查点恢复。**不要为查看缓存而重复启动 `run.py`**，因为它会刷新总运行清单和墙钟时间；首次实测值已归档。完整复现实验应在新的隔离副本中进行，保留冻结历史输入，为该副本准备空的正式结果路径。

`protocol.proposed.json` 是批准前快照；`protocol.approved.json` 是收到“开始”时冻结的协议，其中继承的 `formal_run_executed: false` 描述批准时刻，不是当前状态。当前完成状态以 `run_manifest.json` 与各组 `completion.json` 为准，勿修改协议来覆盖历史。

## 历史测速复现

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

正式报告已按现有八节模版生成，包含 exp004 及更早实验的费用、购电量、预测误差、电池操作强度和耗时可视化。历史测速与预算保留为批准前记录；没有根据评价年成绩重新选择参数或改写 exp001–005。
