# exp007 RL 规划报告契约（正式运行前）

此文件仅冻结报告规范与历史来源，不是正式实验报告，不含 exp007 正式成绩，也不登记实验。先测速、给出单任务与完整矩阵时间估计；必须获得本轮用户明确确认之后，才能开始正式运行。先前 exp006 或 exp005 的批准不转用于本轮。

## 模板来源与同步范围

- 模板基准为 `main` 提交 `7ecd9b988665ad628dbf048ed295457166170ffc`（`docs(reports): require battery power curves and variation diagnostics`）。
- 从该提交逐文件导出全部 Git 跟踪的 `reports/templates/` 文件，并同步 `reports/README.md`、`reports/new_experiment.py`，目标仅为当前 RL 工作树。tree 分支特有额外模板文件保留。
- 本文件不触发预测训练、RL 正式训练、全年评价、报告构建或正式登记。正式运行开始前应再次检查 main 是否有更新；若有，则同步并更新本契约的提交与校验值，不能声称仍是最新而不检查。

| 模板或共享入口 | SHA-256（源提交与同步后文件一致） |
|---|---|
| `reports/templates/battery-power.inline.html` | `b980758e497bcbd889b5ded6182b29c10b97f98ca2479b71ba22c6cfb5669308` |
| `reports/templates/battery-power.md` | `31226bfbdccec7eec858711ace76c03e2ad1e36b929d18a40f071d94da6cc058` |
| `reports/templates/experiment.schema.json` | `f9db28aae84126e99839264de8b9eb0ebdafc91842c23988d13e86838ddab247` |
| `reports/templates/history-comparison.inline.html` | `d2cf95bf4cf00b44c6f204a024e56fb39de4d563b9a1103f79bae1475a929ac9` |
| `reports/templates/history-comparison.md` | `9bc10fd8bb5eef6934c8f2b9049310876309e2271c2fac115f7816f040c5931e` |
| `reports/templates/methods-neural-v1.md` | `9ffeb505b093337f67fdaa0f4587407f97c5e79c30588af53a39228e5c2612fb` |
| `reports/templates/methods-neural-v2.md` | `752eed6c120d0356c7b09e8fe9c6de77490fa5c43caf685e1f7f9a72604311cf` |
| `reports/templates/metrics.inline.html` | `b50e1eb38696461dc877787b953a9ec51c5d02d5f191b7604f68f27bbd2b056b` |
| `reports/templates/q2-discussion/README.md` | `02b4f21231deb7a97234611af9729ef43cd69e6ffb7369681cd718e30531cdbd` |
| `reports/templates/q2-discussion/ReportContent.jsx` | `57e23901af24c30084be55a76c55c57ea737426a0e8edc289cf22bb11e64561e` |
| `reports/templates/q2-discussion/inline-overview.html` | `599b09d5330d7c91dfbf1f7b04e9cd47db852d4a6201610ffdbd90307a07417a` |
| `reports/templates/q2-discussion/report.css` | `4246ef040b49c1483483c5ac41d98b04b2d93e9daadbc00c9833ce5b2b5631fe` |
| `reports/templates/q2-discussion/template.html` | `9660dec046dbebdc4b635e34f010f65c7766a1954fea23be10549b210b52b6a6` |
| `reports/templates/report.md` | `03c218e21c13b5f05419629cdc238b58dc0ef8f7ef39e0dd6855047ea76a5cb4` |
| `reports/templates/technical-path.inline.html` | `6f3f872c60459fe285fc4090489189e8cbf7076386eec11b6cbe1d29b07d7e6f` |
| `reports/README.md` | `864abaaf00e724da443c9e0d3adc43c4d2e7681095c1dbd36c1d3556d2c6dba7` |
| `reports/new_experiment.py` | `fec6283d46b85ff0a0a4887e3aeec655df25a84d250798de7ded19a1f1852518` |

## 报告及机器可读产物

正式完成后在 `reports/experiments/exp007/` 交付八节 Markdown 正文、单文件离线交互 HTML、逐步方法说明、PNG/SVG 论文图、指定四日完整表、基于原题模板的 `result2.xlsx`，并保留 `app/` 网页源码与完整数据快照。八节依次为结论与成绩、指标与信息边界、数据与时间验证、技术讲解、实验设置、结果与失败、历次比较、复现说明。

`evidence/` 应保存完整费用/预测/电池/阶段耗时 CSV、逐日和逐槽执行来源、相对变化、历史来源和协议差异、种子成绩及均值/样本标准差、输入和代码 SHA-256、冻结配置、计时记录、独立核验与浏览器视觉验收。正式登记记录依 `reports/templates/experiment.schema.json` 填入真实值；运行源码必须由完整 Git 提交与逐文件哈希绑定。不得覆盖 exp001–exp006 的注册表和旧报告。

原始执行接口优先兼容既有 `dispatch_2.npz`：计划购电 `original`、最终普通购电 `final`、实际 `charge/discharge/emergency/surplus`、`states`、分项 `fees`、`actual` 和 `price`，另存 RL 动作及裁剪信息。明确每个字段单位、维度、时间轴和方案角色。题目费用仅由实测计划购电与紧急购电结算；奖励中的吞吐、反转、平滑偏好单列，不能混入题目总费用或称为已标定折旧。

第 4–5 节须解释实际 RL 的状态、动作、奖励、策略/价值网络、参数更新、训练样本时序边界、模型选择规则、训练与评估种子及训练预算。方案角色在正式评价前冻结；不能根据全年费用替换正式策略。复用 exp004 冻结预测时，预测误差必须与其对应家族/种子一致，不能把规划变化表述为预测精度提升。

## 主目标、约束及评价口径

总费用优先；互斥、真实电池容量/效率/功率/SOC 及因果信息边界由实际环境和独立核验保证。减少大量充放、方向切换及功率波动属于次要偏好，分别报告收益和费用代价，不能只展示有利指标。电芯等效全循环、交流侧吞吐及反转次数只描述运行强度，没有化学体系、温度和寿命资料时不推算寿命延长。

继承历史同口径比较时，必须逐项验证：2025-02-01 至 2025-12-31 的 334 日、每日 144 个十分钟槽；午夜购电固定、`original=final`；费用 `sum(price * planned_kwh + 5 * price * emergency_kwh)`；无期末残值抵扣；容量 12000 kWh、SOC 1200–10800 kWh、单向效率 sqrt(0.9)、单向功率上限 5000 kW；共同预热末 SOC 1421.7991105135516 kWh 与上一槽净功率 172.76 kW。各策略应连续承接自己的实际跨日状态。这些是历史参考口径，不意味着本轮未实现方案已经通过核验。

## 强制电池曲线与波动证据

以 main 的 `reports/templates/battery-power.md` 为完整要求，第 6 节及最终对话必须落地以下内容：

1. 实际母线侧净功率 `P=(charge-discharge)/Δt`，充正放负，单位 kW；十分钟 kWh 乘 6。另存非负充、放功率。SOC 差分通过 `SOC_next-SOC=η_charge*charge-discharge/η_discharge` 独立验证，不能把 SOC 差当母线功率。
2. 固定随机种子默认 20260912，从完整正式评价日不放回抽取至少四日，排序并记录实际日期、算法和种子；所有方案使用相同日期与纵轴。保留原始 144 点、突变、零线与功率上限，直线连接，不做平滑或插值。说明区间端点：00:10 表示 00:00–00:10 的终点。
3. 全评价曲线包括每方案全部 48,096 个原始时点，标为 334 日，不能称为新策略完整 365 日成绩。只有真实、连续的预热档案才能另拼 52,560 点的 365 日图，并区分预热/正式期，分别核算。随机日与完整评价均交付独立 PNG/SVG 和原始 CSV，交互图可缩放。
4. 连续时间轴的 `ΔP` 包含跨午夜相邻槽，评价首点无差分，不能补零；预热末点至正式首点跳变单独报告。至少给出平均绝对变化、RMS、绝对变化 P95、最大绝对变化、总变差、相邻槽直接充放反转次数、功率贴限比例，并记录容差与计量单位。
5. 大跳变阈值预先声明，默认 `|ΔP|>1000 kW/10min` 仅为描述阈值，不能误写成题目硬约束。若采用真实爬坡约束，必须单列其数值、来源及对历史可比性的影响。
6. 区分“相邻两槽直接反转”和“忽略空闲槽后的非空方向反转”；exp006 原指标属于后者，不能直接改名充当新模板前者。既有 `power_variation_kw` 只有总变差，缺失的新指标须从冻结逐槽档案核算。
7. 时间完整性、有限值、非负分量、互斥、功率/SOC 边界、预热与跨日衔接均独立核验。没有档案不能填零。对比输出原值、绝对差和相对变化，基线为零时相对值为空。

`battery-power.inline.html` 的 `__BATTERY_EVIDENCE__` 只作为实测数据插槽；必须绑定已核验数据。该模板目前只提供随机日视图，不能代替全年图和完整波动指标。最终对话至少实际展示随机日曲线。

## 历史 exp001–exp006 的来源与可比性

| 实验 | 冻结来源及角色 | 比较处理 |
|---|---|---|
| exp001 | 当前 RL/tree 工作树的 `reports/registry/exp001.json`；exp006 evidence 保留 `exp001/original` 与 `exp001/legacy_rebased` | 原登记物理口径不同，仅并列说明；同物理重算可另列，必须保留重算来源。exp001/2 既有共享四目标早停边界照实说明。 |
| exp002 | `reports/registry/exp002.json`；exp006 evidence 的 `exp002/primary`（正式风险）及 `exp002/new_deterministic`（历史确定性对照） | 正式与对照角色分开。费用、预测、耗时各自审核口径，不把不同控制器的差异都归因预测。 |
| exp003 | `reports/registry/exp003.json`；`data/results/exp003/`；exp006 evidence 的 `exp003/primary` 和 `exp003/uncalibrated` | 独立问题二正式/未校准对照；保留共同预热及逐槽来源。 |
| exp004 | `reports/registry/exp004.json`；`data/results/exp004/predictions.npz` 与各家族 `dispatch_2.npz` | `no_season` 是 exp006 同预测主基线；`causal_season` 是该轮正式历史季节；`oracle_season` 含未来，仅探索且不排名。 |
| exp005 | 根工作树 `/Users/vegetarianwolf/Projects/CUMCM2026/reports/registry/exp005.json`，原始数组 `data/results/exp005/soft-penalty-beta-0.1/dispatch.npz` 与 `soft-penalty-beta-0.01/dispatch.npz`；报告 `reports/experiments/exp005/report.md` | 当前已登记 β=0.1 主方案和 β=0.01 对照，均记载完成 334 日。额外硬爬坡 1000 kW/10min，LP 连续松弛互斥、原执行允许紧急购电服务充电，控制口径不同。当前尚未对其原始数组、时标、计费及输入数值完成本轮独立兼容性审核；先保留为待核对描述性历史，不能直接排名或生成改善率。 |
| exp006 | `reports/registry/exp006.json`；`data/results/exp006/`；`reports/experiments/exp006/evidence/` | `primary` 正式树DP与意图裁剪执行；`greedy_execution` 保留已测消融角色；`fixed_primary_greedy` 保留执行控制角色。不能因消融费用更低，事后把它改为 exp006 正式。 |

exp006 历史证据接口可复用：`evidence/cost_history.csv`、`forecast_history.csv`、`battery_history.csv`、`relative_comparison.csv`、`timings.csv`。这些来自冻结档案的证据生成器在 `reports/exp006_evidence.py`；报告入口为 `reports/build_report_exp006.py`，对话图数据绑定为 `reports/build_inline_exp006.py`。这些文件针对 exp006 硬编码路径/角色/15 组，不能原样运行来生成 exp007，更不能通过重建它们改写旧报告。新实现需要针对 exp007 生成新的证据与报告，读取旧证据而不重写。

exp006 的 `registry_coverage.json` 只列 exp001–exp004，因为其当时尚未纳入 exp005；exp007 必须逐一列出 exp001–exp006 的实际读取来源、哈希及是否纳入原因。不能沿用 exp006 的“exp005尚未完成”陈述。exp005 注册表未采用通用完整 schema，缺少 `technical_path`，其 `metrics` 行缺少 `scenario`；原样调用 `reports/history.py` 会失败。应在 exp007 自己的证据层做显式、可审计的适配，不回写旧登记。旧记录缺失字段标为空。

exp005 当前全年软惩罚证据位于根工作树 `reports/experiments/exp005/soft_penalty/evidence/`，其中 `period.csv`、`runs.csv`、`actual.csv`、`audit.csv`、`timing_history.csv`、`forecast_history.csv` 可作为后续审核入口。不要误读其同级旧 `reports/experiments/exp005/evidence/independent_audit.json` 和 `report_qa.json` 为全年主方案证明：它们仍明确对应早期 β=0、81 槽停止的失败案例。本契约只定位来源，尚未执行 exp005 全年重新核验。

exp005 记录的原始 CSV 字节哈希与 exp006 不同，冻结预测档案记录哈希相同（`6f7bcf27e420e2f05251820a5c1d8d1f87cd6234287e23505b33a51ae6d757b9`）；这只是已有记录的声明，不能代替本轮文件/数值/时标核对。未核对前不能假定仅换行或编码差异。

## 历史可视化与正式核验

第 7 节及最终对话应从同一批核验 CSV/JSON 展示费用分项、WAPE 与 RMSE（完整数据保留 MAE；全时段/实际发电时段分开）、阶段耗时、技术路线。预测复用时显式说明没有新增预测训练。RL 训练时间与预测网络训练时间分开，特征/环境准备、策略训练、验证选择、策略推理、逐槽执行、导出、报告分别计时；累计求解时间与墙钟有包含关系，不能相加。不同机器和历史计时缺失仅作描述，缺失不填零。

可比较数据保留 `previous`、`current`、`absolute_change`、`relative_change_pct=100*(current-previous)/abs(previous)`、来源、正式种子、评价范围和可比原因。WAPE 原值为百分比，绝对差为百分点。不可比的数据不画改善率/连接线/排名。报告、工作簿、PNG/SVG 和最终 Visualize 图必须同源，负面结果和失败项也保留。

正式交付前完成独立费用、能量平衡、SOC/功率、互斥、连续时间边界、因果输入及原题工作簿读回核验，并进行桌面/窄屏浏览器视觉和交互检查。网页构建成功不等于浏览器验收完成。最终对话必须使用 Visualize 实际展示历史费用与误差比较，以及随机日实际电池曲线，不能只有链接。
