# exp003：第二问独立预测与费用校准

分支 `codex/q2-optimization` 来自 v2 提交 `382197175a19a044c871f27d73999715fe81f1be`。
本轮只优化第二问；旧 v1/v2 代码、结果及报告的 906 个文件有冻结校验。

## 冻结选择与实测结论

- 用户明确选择 D001–D003=A：充放各 sqrt(0.9)、轻量预测加费用校准、1月25–31日共同调参。
- 第二轮用户回复“继续”，按已说明的推荐默认项 D004–D006=A：四时点训练/午夜早停、确定性日前调度、两目标各取 {0,0.5,1} 的九组系数。
- 所有备选路线和依据先写入 [决策日志](decision-log.md)，决定和实测结果随后追加。

九候选按一月七日费用选中 **α负载=0、α光伏=0**，此后全年冻结。这说明本次费用准则拒绝网络残差，不能宣称神经网络带来费用优势。正式费用 14,988,622.6482 元；未校准网络 15,258,030.0912 元；原始周期基线 14,988,614.2346 元。正式方案略高于周期的 8.4136 元涉及 float32 基线和求解数值变化，不视为收益。三个种子的正式预测相同，零费用方差不构成网络独立稳定性的证据。

相比 v2 正式风险方案，总费用下降约 0.927%，日费用 CVaR90 上升约 2.111%。报告分别展示历史确定性对照、Q2 独立未校准模型和校准后方案，不把调度变化全部归因于预测。

## 边界与复现

读取/哈希白名单只有附件1及附件2两张表。每月从头训练负载/PV两分支 16→32→16→1 MLP；共 11 月×3种子=33组，每组2178参数。标准化、特征、早停及系数选择均遵守历史标签截止。1月25–31日同时用于早停与费用选择，是共同调参段；全年结论为回顾性时序评估。

正式评价覆盖2025-02-01至12-31，共334天、48,096个午夜发布目标。一月从6000kWh连续预热，全年SOC接续；日前计划全天固定，紧急购电按5倍固定电价另付。执行器保持 v2 因果贪心规则。

从仓库根目录执行，`NODE` 指向 Codex 捆绑的 Node 可执行文件：

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_q2_*.py' -v
.venv/bin/python -m experiments.problem2.exp003.baseline_evidence
.venv/bin/python -m experiments.problem2.exp003.provenance verify

# 需要可用的 Metal GPU；代码、配置、数据和权重哈希会核验已有缓存。
.venv/bin/python -m experiments.problem2.exp003.run train
.venv/bin/python -m experiments.problem2.exp003.run predict
.venv/bin/python -m experiments.problem2.exp003.run evaluate
.venv/bin/python -m experiments.problem2.exp003.run verify
.venv/bin/python reports/q2_comparison.py
.venv/bin/python -m experiments.problem2.exp003.run export --node "$NODE"
.venv/bin/python -m experiments.problem2.exp003.run report --node "$NODE"
```

`run all --node "$NODE"` 顺序执行完整流水线。训练权重与逐日缓存位于忽略的 `runs/`，全新克隆需重训后才可执行完整核验；数值库和设备的非确定性可能导致重训差异。报告构建只消费已保存结果，不需训练权重。年度 `predictions.npz` 的 float32 快照不是正式评分入口，评价统一用 `ForecastStore.get` 的 float64 混合值，并保存 `evaluation_predictions.npz`。

## 交付

[八节报告](../../../reports/experiments/exp003/report.md) · [离线交互报告](../../../reports/experiments/exp003/report.html) · [第二问工作簿](../../../reports/experiments/exp003/result2.xlsx)

正文聚焦第二问，完整原理、九候选、指定日期、三种子及历史原始值保留为附件。`data/results/exp003/` 保存完整指标、预测、调度和核验；`reports/experiments/exp003/` 包含可独立阅读的报告、PNG/SVG图和工作簿。
