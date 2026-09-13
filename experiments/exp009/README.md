# exp009：固定 exp008 的费用目标对照

从 `b2ce6b6e4eef46512eb1092c408cef155a37f187` 新建
`codex/exp009-cost-only-control`。用户要求保留模型方法，只去掉运行强度偏好，完成报告后推送该分支。

`protocol.json` 在全年运行前冻结。每问一个候选，不以实际费用或电池次数筛选结果。
Q1只执行原可行域的费用阶段；Q2取消每日8次模式变化上限并将吞吐罚归零；
Q3/4-3将吞吐/变化罚与充电死区归零；Q4-2保持小时模式，把换向/吞吐罚归零。
预测信息、数据、电池物理、初始状态、场景构造、求解预算和经济终值均保持。
Q3/4-3按原代码重新计算既定在线岭校准，发布输出与exp008逐哈希一致；无新树模型训练。

所有年度运行完整334日，原始轨迹在 `data/results/exp009`，正式报告与五份模板工作簿在
`reports/experiments/exp009`。Q2费用略升，其他问费用小幅下降，年度换向和吞吐均增加。
全部结果保留，不继承上一轮费用/次数下降门槛。详见报告中的同口径分项和限制。

执行入口：`run_q1.py`、`run_q2.py`、`run_update_cost_only.py`、`q4_2_cost_only.py`。
这些入口拒绝覆盖已冻结结果。用新检出或入口支持的独立输出目录重新实验；有时间预算的MIP跨机器不承诺逐位复现。

仅读取冻结档案重建：

```bash
.venv/bin/python -m experiments.exp009.final_evidence --require-all
.venv/bin/python -m experiments.exp009.report_payload
.venv/bin/python -m reports.build_report_exp009 --code-commit <record.json中的40位源码提交>
.venv/bin/python -m experiments.exp009.inline_comparison
.venv/bin/python -m experiments.exp009.delivery_check --require-committed-source
```

最后一项需要工作簿、浏览器QA已就绪，工作簿命令见报告的 `evidence/workbooks/README.md`。
各问的原始数据、真实账单、电池物理、预测/历史身份、运行源代码闭包与原exp008档案保持分别审计。
本实验不宣称真实年度费用全局最优、严格非预见性场景树最优或电池寿命收益。
