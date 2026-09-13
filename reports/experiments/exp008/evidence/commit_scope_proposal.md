# exp008 提交范围与体积建议

本文件仅记录提交建议；没有删除文件、修改 ignore、暂存或提交。文件大小为本次检查时的未压缩字节，不是 Git 仓库压缩后的大小。

当前 data/results/exp008：2114.523 MiB / 11205 文件；建议保留 118.043 MiB / 2094 文件，减少 94.42%。

加上完整报告、五份工作簿、CSV/PNG/SVG 证据与代码测试，本次建议新增/修改的文件合计 262.643 MiB。已被 Git 跟踪的旧实验和原始题目附件继续保留，不重复复制。

## 保留范围

- 最终 Q1 结果；Q2 的实际执行、全部 334 日规划 NPZ、发出的预测、协议与因果/物理核验；Q3/Q4-3 的完整年度结果和全部发布时点审计；Q4-2 的完整实际结果、原始及勘误后审计、逐日规划证据。
- canonical HGB、ExtraTrees、blend、价格预测的冻结 NPZ 和审计/配置，保留原始→Ridge→memory 三阶段预测。受控比较保留 HGB 同约束实际调度、两个 LP bridge 的实际调度和摘要。
- 所有未选候选的 summary.json 清单以及协议、配置、来源和失败/停止记录；不保留它们的逐日决策数组和大体积调参缓存。现有 net-HGB 单元测试读取的四个冻结 fixture 明确保留。
- reports/experiments/exp008 下完整交付成果及证据，包括 13 组全部 48,096 时点 CSV、统一随机四日与指定四日、全部 PNG/SVG。exp005 两份冻结来源快照和锁定 main 的三份模板必须保留。

## 可排除范围

| 类别 | 文件数 | MiB |
|---|---:|---:|
| exclude_duplicate_forecast_archive | 305 | 83.033 |
| exclude_duplicate_workbook_copy | 5 | 3.234 |
| exclude_model_weights_and_runtime_cache | 526 | 856.361 |
| exclude_unselected_arrays_and_intermediate_diagnostics | 8271 | 925.487 |
| exclude_workbook_inspection_dump | 4 | 128.365 |

所有文件的精确相对路径和保留数据 SHA-256 位于相邻 JSON 的 keep_files / exclude_files；existing_tracked_dependencies 是已跟踪依赖，不是再次新增列表。

## 重建与审计边界

报告重建、实际费用/电池原始方程复核、冻结预测评分均不加载 joblib。完整 final_forecast_evidence.run() 还会核验原来的 44 个模型权重文件；这属于可选权重级历史审计，不属于本提交范围的报告重建命令。保留其已完成的模型哈希和训练边界记录，预测图可直接从已提交 CSV 重新绘制。训练重跑应使用新的输出目录，不能覆盖冻结实验，且有时间上限的 MIP 不承诺跨机器逐位相同。

新 clone 先重建 final_payload 到当前 checkout 路径，再做 payload 原始数组审计和 report build。现有载荷中的绝对路径是原执行时来源；从 manifest 重建后会产生新 checkout 引用，工作簿数值不变。Git 固定源不可用时，模板与 exp005 使用显式且 SHA 锁定的快照 fallback，不再依赖本地 main 或侧分支。

```sh
.venv/bin/python -m experiments.exp008.refresh_report_payload
.venv/bin/python -m experiments.exp008.final_payload_source_audit
.venv/bin/python -m experiments.exp008.final_forecast_independent_audit
.venv/bin/python -m experiments.exp008.final_battery_evidence_audit
.venv/bin/python -c "from experiments.exp008.final_forecast_evidence import plot_figures; plot_figures()"
```

parent 仍在刷新最终报告与 source hash 元数据，因此本提案不绑定报告 JSON 的旧哈希；冻结 NPZ 的数据哈希列出以供暂存前检查。所有未选候选只能按保存的范围描述，停止候选不可推断为全年完成。
