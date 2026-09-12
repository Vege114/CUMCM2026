# exp002 结果数据

固定代码版本见报告的 `record.draft.json` 与注册记录；第一问沿用原成果。正式种子为 42，2026、3407 是独立稳定性重复，不能按成绩重新选择种子。

- `predictions.npz`：`origins` 为从 2025-01-01 00:00 起计的十分钟发布索引；`seed_42`、`seed_2026`、`seed_3407` 形状为发布数×144×4，变量顺序为负载、历史光伏、电价、光伏预报修正。目标区间为 `[origin+h, origin+h+1)`。功率单位 kW，价格元/kWh。模型身份、月度训练边界与检查点指纹见 `prediction_archive.json`。
- `forecast_metrics.csv`：月份、种子、变量、发电时段与提前量的误差充分统计量；`lead=all` 与各提前量分组不可重复累计。跨年缺少真实值的目标不纳入误差。
- `seed_forecast_results.csv`、`seed_forecast_statistics.csv`：三个种子的各自全年预测误差及均值、样本标准差（ddof=1）。统计的是三组指标，不平均预测；WAPE 标准差的单位为百分点。
- `january_baselines.csv`：正式评价之前的一月历史同期诊断，93 个完整发布窗口；仅使用一月标签，支持固定周期组合的选择依据。
- `dispatch_*.npz`：四个正式问题的全年逐区间原计划、最终计划、实际供需、储能执行、电价、四项费用与 145 个状态节点；前两维为 334 天×144 区间（状态为 145）。电量单位 kWh，费用元。
- `warmup_*.npz`：一月共同因果基线预热。2 月 1 日初始状态来自对应预热的最后状态，全年保持连续。
- `daily_metrics.csv`、`dispatch_metrics.csv`：所有具名策略的逐日及全年费用；核心比较必须在同一问题、同一物理和结算协议下进行。
- `solver_metrics.csv`：三个正式种子的全部风险调用，记录信息截止时刻、场景历史、求解状态、可行来源、约束残差及全局差距。空差距代表未取得证书，超时与回退分别统计。
- `risk_calibration.json`：仅以一月七日回放选择风险权重。三个候选各自的日费用和选择时刻可检查。
- `evaluation_manifest.json`：全部策略配置、缓存键、代码与上游签名。零风险权重与正式配置相同时共用缓存，不能把重复命名当作独立重复实验。
- `official_forecast_comparison.csv`：按旧正式策略读取 exp001 月度选择；涉及光伏预报的旧十分钟预测点按本轮积分口径转换后单列重算，旧档案及注册记录不变。
- `specified_dates.csv`、`specified_intervals.csv`、`battery_blocks.csv`、`emergency_periods.csv`：题目表格和工作簿的同源未舍入数据。
- `failure_intervals.csv`：各问题最贵日期的 144 个区间；`hour` 是区间中点，`soc` 为该区间结束状态。绘制储能轨迹使用 `failure_storage.csv` 的 145 个实际状态时刻，完整覆盖 00:00—24:00。
- `verification.json`、`full_year_audit.json`、`model_checks.json`、`prediction_recovery_check.json`：独立物理/计费核验、33 组 GPU 训练检查和从权重恢复预测的检查证据。
- `calibration_causality_check.json`：扰动二月以后数据后，一月正式检查点对应的 70 个校准决策时刻保持输入不变。
- `report_archive_independence_check.json`：报告改用已交付预测档案后，样本、分解曲线、场景树及历史预测比较完全一致；重建正文和图表不需要私有训练缓存。
- `causal_tree_fix_migration.json`：固定电价问题的信息隔离修正。六组受影响的初步风险回放及问题 3 校准作废重算；其余缓存按明确的代码影响范围和逐月等价性检查保留。它是公开记录的兼容性迁移，不是忽略代码签名变化。
- `clipped_information_fix_migration.json`：信息前缀改用裁剪后的实际场景。逐个检查已有回放的输入，只保留完全等价前缀，并作废受影响后缀及校准。`known_price_information_migration.json` 进一步记录已知价格反事实的信息隔离修正。

四个 `result*.xlsx` 由官方模板导出。全天费用包含原计划费、最终净调整费和紧急购电费；工作表中重复列出的全天总费不能再次相加。输出物理口径、结算解释及缓存签名见 `protocol.json`。阶段时间必须连同 `phase_timing.csv` 中的计时范围阅读，任务累计时间与墙钟时间不能直接相加。

`reports/experiments/exp002/` 复制浏览报告需要的 JSON、CSV 和工作簿，离线 HTML 完整内嵌交互数据。体积较大的逐区间 NPZ 留在本实验分支此目录，训练权重与可恢复逐日缓存留在本机被 Git 忽略的 `experiments/common/neural_v2/runs/exp002/`。
