# 独立前置核对

核对人：独立只读验收子 Agent；未参与被审代码、正文或结果的编写。核对日期：2026-09-13。

## 结论

35 项人工决定已完整阅读。F10 明确选择退款解释并重新优化，F12 明确将冷启动接入最终策略贯穿全年；这两项是对清单开头“冻结原 exp008 数字”的明确例外。后续应以 exp008 派生结果为正式口径，原 334 日费用只能保留为有标签的历史对照。

P1/P2/W1/W2 均尚未验收；本文件仅为前置核对，不是通过证明。

## 官方证据

- 竞赛时间为 2026-09-10 18:00 至 2026-09-13 20:00（北京时间）。[官方国际站](https://en.mcm.edu.cn/)、[官方首次通知页](https://www.mcm.edu.cn/html_cn/node/d6fd7a0ee8f3a3d525e30af1c365fcec.html)、[官方通知PDF](https://www.mcm.edu.cn/upload_cn/node/779/D3txF20S95f4041e22b41b4a1f9e0e22ac8a1389.pdf)。已核实竞赛截止时间，未将高校通知中的上传时间当成统一官方事实。
- [2026 年论文格式规范](https://www.mcm.edu.cn/html_cn/node/4cd596519c9eb9fbd866398f6df0caa3.html)：A4，页边距至少 2.5cm，摘要原则上一页，正文不超过 30 页，附录包括完整可运行源程序与支撑文件列表；电子论文 PDF/Word 不超过 20MB，支撑 ZIP/RAR 不超过 20MB。电子论文以摘要页起始，不含承诺书与编号页。
- [2026 年 AI 规定](https://www.mcm.edu.cn/html_cn/node/fef94648f2836ab6cc81586f4c38512b.html)：参考文献前放 AI 声明，支撑材料包含“AI工具使用详情.pdf”，说明工具/版本、用途、提示过程和采纳/人工修改/核验情况。
- F33 人工备注明确本轮暂缓页数与既有约 289dpi 优化；该范围应据实记录，不自行扩展为排版重构。不能因此声称已符合正式提交全部格式要求。

## 恢复范围

对冻结提交 `459cdb2` 的 Q1、正式 Q2、Q3/Q4-3、Q4-2 四入口做静态 AST 依赖闭包，得到 39 个本地 Python 文件。exp008 之外的必要模块包括：

- experiments/common/neural_v2/data.py
- experiments/common/neural_v2/physics.py
- experiments/problem2/exp003/data.py
- experiments/problem2/exp004/data.py
- experiments/problem2/exp004/predict.py
- experiments/problem2/tree_planning/{__init__,model,risk,verify}.py

非标准库依赖为 numpy、scipy、pandas、scikit-learn、joblib、threadpoolctl、openpyxl、matplotlib。入口闭包没有直接 TensorFlow 依赖；保持冻结预测缓存即可避免不必要训练。

## 必需的实现检查

1. 从 day=0、SOC=6000kWh 跑满 365 日；不能加载 run.initial_state 中的 exp002/warmup 档案作为二月初态。Q2 计划末模式及已保持槽数必须连续传递。
2. Forecasts._midnight 和 AbsoluteHGB/Blend 适配器已有 day<31 的因果周期冷启动；退款和储能初态改变不影响既有供需预测，可复用相同冻结预测缓存，同时如实说明没有重训。
3. 原 planner.plan 通过 q>=g 限制只上调。可保持既有 LP，用 q>=0、a+/a->=0、q-a++a-=g，目标增加 1.5p*a+ - 0.5p*a-。由于正负偏差同时增加的净成本为 p>0，最优解自然不同时上/下调。实际账单及独立核验必须同步 down_cost 负号。
4. `Forecasts.net_error_paths` 和 `issued_error_paths` 不支持 day=0，day=1 无历史也会失败。无已完成历史时应提供零误差路径并留空真实历史 origins，随后纳入合法已完成日。不能把虚构零残差称为历史实测。
5. `Forecasts._price` 与 `LinkedPriceForecasts._price` 在 origin=144 时训练集合为空，但直接读取 train[0]；必须加不足历史的冷启动分支。只在子类回退 super 无法修复同一个基类错误。
6. 新轨迹通过 P1 后才能全量运行；验收至少覆盖第 0/1 日、第一次日内更新、第 31 日缓存切换，以及 12 月 31 日末库存估值归零。
7. 365 日新主轨迹必须统一所有指定日/区间表、摘要、年度费用及冷启动初态。附件5要求工作簿仅保存2月1日至12月31日334日，因此从新轨迹[31:]导出原模板；可同时列365日与334日费用但需明确。旧334日预测误差如保留须明确采样窗口，不能伪称新全年值。

## 预算证据

冻结 exp008 计时：Q2 334 日 MIP 求解合计约 699.1s，固定模式购电修正约 304.3s；Q4-2 分别约 934.0s、342.3s。Q3、Q4-3 的原 LP 全部运行墙钟约 76.2s、90.7s。它们只用于运行量级判断，不能当成本轮实际耗时或相加宣称总墙钟。

## Skill 完整性

原安装目录缺少子入口，主 Agent 已从其 Git 恢复完整 Skill 到 tmp/conflict-resolution/skill/。本验收 Agent 实际读取了恢复副本的 references/Subagent调度.md、references/roles/编程手/SKILL.md、references/roles/论文手/SKILL.md、references/交付与截止时间协议.md，以及论文手的写作规范和自审框架。缺失入口限制已解除；任务依用户明确范围仅修改和编译当前 TeX，不增加 Word 或初始化新论文目录。
