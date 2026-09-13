# exp008 技术数据接口（尚未选定最终模型）

模板读取当前 `main` 的 `reports/templates/report.md`、`battery-power.md`、`history-comparison.md`，每次输出记录实际提交与文件 SHA256。当前检查提交为 `7ecd9b988665ad628dbf048ed295457166170ffc`。本模块只写技术 JSON；不写报告、HTML、Word、图片或 Excel，不自动选择费用最低档案。

调用：

```sh
.venv/bin/python -m experiments.exp008.report_payload experiments/exp008/report_manifest.unselected.json --output data/results/exp008/report_preparation/unselected_payload.json
.venv/bin/python -m unittest experiments.exp008.test_report_payload
```

复制未选择的 manifest，明确填写每个场景来源，使用新的输出路径。`--mode final` 要求全部材料并独立核验 Q2 完整 334 日的费用至少下降 8%、非空方向反转少于 2729、零同充放与物理一致性，否则抛出 `FinalPayloadRejected`，不写最终 JSON。费用门槛由 `0.92 * 14066257.477256786` 计算，不能换对照或让其他指标补偿。即使通过，也仅表示可进入最终报告制作，不代表报告/工作簿已经完成。

## 输入 manifest

`q1.archive` 明确指定 Q1 两层解 JSON；必须有 stage 1 和 stage 2、最终 stage 2 轨迹、config、budget。加载后重算附件 1 账单，并运行两层约束与通用物理核验。

`scenarios` 的键是 `2`、`3`、`4-2`、`4-3`，每项包含：

| 字段 | 含义 |
|---|---|
| archive | 已核验 dispatch NPZ 的显式路径，再次对原始附件独立核验 |
| role | formal primary/development/control/ablation 等明确角色，不从费用推断 |
| start_day | 以 2025-01-01 为第 0 日；正式评价为 31 |
| initial_soc_kwh | 独立核验的暖启动末状态，不能从候选首槽反向认定；Q2 固定 1421.7991105135516，Q4-2 固定 1390.382746315672，其他场景显式填自己的暖启动来源 |
| initial_mode / initial_power_kw | 暖启动末模式/功率；若缺省沿用通用核验默认值，仅影响边界诊断，正式内部换向不含此边界 |
| audit | 每个发布时刻的 origin / max_observed_index / 预测来源等 JSON |
| causality_verification | 独立未来扰动检验等证据路径；文件需由最终审阅者检查作用范围，存在文件本身并不证明因果性 |
| eligible_as_causal_strategy | 明确为 true 才允许 final；事后最优/完美未来执行下界不得标为可部署策略 |
| forecast_sources | 结构化来源：base_model、seed、correction、risk_history、price_model、information_cutoff、source artifact/hash；逐场景声明，不能用共享标签掩盖不同版本 |
| power_manifest | 已由 power_diagnostics.py 生成的 manifest.json；必须对应本档案 SHA256 和场景 |

`prediction_evidence` 明确提供 annual_metrics、monthly_metrics、lead_metrics、provenance、model_configuration、verification 文件。内容应涵盖 load/PV/net/动态 price、原始单位、误差充分统计、光伏活跃时段、发布提前量、最新网络各层和训练信息边界、校准公式与来源。JSON 仅固定这些文件的路径、SHA256 和字节数，不把未经确认的模型版本拼成“最终预测”。

`supporting_evidence` 的 runtime_by_stage、seed_and_ablation_results、planning_solver_diagnostics、data_validation、reproduction 同样用明确文件路径。包括训练/预测/风险/规划/执行/导出/报告时间，硬件与复用范围，样本标准差，多层目标/软约束预算、MILP gap/限时/回退情况，变量约束规模，附件质量与完整复现命令、源码/data hash。未知 gap 与不可比时间保留 null/说明。

## 统一档案及输出

| 数组 | shape | 单位/语义 |
|---|---|---|
| original, final | N×144 | 每槽 AC kWh；分别为冻结原计划、最终绝对购电量，final 不是增减量 |
| charge, discharge, emergency, surplus | N×144 | 真实发生的每槽 AC kWh，不是控制器理想动作 |
| states | N×145 | kWh，跨日连续 |
| price | N×144 | 元/kWh；Q4 为结算实际价格，优化预测价格必须另外提供来源 |
| fees | N×144×4 | 元，顺序为计划、上调、下调、紧急；不含软惩罚或末端价值抵扣 |
| actual | N×144×≥2 | 前两通道 load/PV kW，额外通道保留原档案定义 |

输出包含每场景：源路径/hash、原始附件对照核验、物理/结算检查、独立重算分项费、实际电池指标、逐日/月加总、最高费用日、指定日期表 1/2/3、五份工作簿所需数据、全轨迹数组位置、已有功率证据引用。问题 1 是附件 1 的确定性给定日，不能称为真实观测全年样本。

题面指定日期为 2025-03-20、06-21、09-23、12-21。表 1 的起点为 10:00、12:00、14:00、16:00、18:00、20:00，对应槽 60/72/84/96/108/120；表 2 聚合六个连续 4 小时时段，并列 00:00、24:00 SOC；表 3 将同日连续正紧急量合并，保留结束 24:00 的事件，无事件标“无”。

Q1 的原始 `trajectory.c/d` 是 kW，转换为 kWh 时除以 6 一次；输出保留功率与电量两个显式字段。其他四问 `charge/discharge` 已经是 kWh，不能再除以 6。

## 五份工作簿的数据契约与已有导出器缺项

`result1.xlsx`：`计划购电量` 的 144 行 `[interval,purchase_kwh]`，`充放电量` 的 6 行 `[interval,charge_kwh,discharge_kwh,soc_label,soc]`。现有 `reports/export_workbooks_v2.mjs` 不支持 Q1，需要后续单独 adapter。

`result2/3/4-2/4-3.xlsx`：兼容现有导出器的 dates、original、final、fees、battery、emergency；另提供 interval_headers、daily_fee_components 和字段语义。完整年有 334 行计划、2004 行电池块、可变行紧急事件；3/4-3 增加最终调整购电页。原模板示例行需完整替换/扩展。

所有模板均存在首时段写成 `0:10-0:20` 的错位，必须替换为 `00:00-00:10` 至 `23:50-24:00`，与附件右端点时间一致。现有导出器把同一日总账单写在原计划与调整计划两张表，不能再跨这两个重复金额求和；技术数据另保留真实费用分项。现有 `neural_v2/export.py.independent_check` 强制无限制 greedy 充放，不能用于共同模式/死区等 exp008 执行策略；本接口已经使用 `exp008.verify` 代替。

## 模板图表来源与缺项

电池图直接引用 `power_diagnostics.py` 的全原始功率 CSV、seed=20260912 的四个无放回随机日、metrics、manifest。全年度图必须使用全部 48096 原始点，同一比较组使用相同随机日，不能平滑或降采样。若没有对应暖启动全年轨迹，只标“334 日正式评价”，不写 365 日。正式首点的 ΔP 无前驱，暖启动边界另列；直接相邻换向与 exp006 的非空方向反转必须分别报告。

历史 exp001–006 全部保留 registry 原始 metrics、预测 metrics、模型/环境/协议及角色，同时另列 exp008 已审计的同口径重算行。exp005 从 baselines.json 记录的 git 源和 hash 读取并校验；其额外硬爬坡和允许紧急购电充电限制要保留，不参与物理同口径排名。exp001 原始效率、exp001/2 原始四目标早停、oracle 信息边界、各实验问题覆盖范围不能省略。忽略 exp007 是用户明确要求。预测可比性与费用可比性分开；各行没有明确比较许可时不做优劣排名。

接口不生成技术流程/网络结构图、预测误差图、历史对比图、功率图、报告或 xlsx。上述图均需后续从同一技术数据源制作独立 PNG/SVG 并检查显示；缺失的明确来源汇总在 `missing_items`。`prepare` 可以完成这份缺项清单，不能将其作为已完成优化的报告。
