# exp006 第二问替代规划路径

本目录复用既有八节报告模板及 Data 报告运行时，包含正式实验、全部种子/消融/细网格、历次同口径比较与可复核证据。

- [完整交互报告](report.html)：单文件离线版；运行时预览从 `app/dist/` 提供。
- [Markdown报告](report.md) 与 [逐步技术说明](methods.md)。
- [冻结正式主组工作簿](result2.xlsx) 与 [指定日期表](specified_dates.md)。
- [费用较低已测对照工作簿](greedy_execution/result2.xlsx) 与 [对应指定日期表](greedy_execution/specified_dates.md)。
- `figures/`：同源 PNG/SVG 论文图；`evidence/`：完整CSV、核验、冻结配置和清单。
- [测速记录](preflight.md)：用户批准前的独立测速，不能替代正式结果。
- [正式登记](record.json)：绑定已验证的正式源码提交 `2bda040f3818d1b4c2a3a3789f0ac58dc22af129`；`record.draft.json` 保留登记前草稿。

正式角色固定为 `primary`（无季节预测seed42、50kWh网格、意图裁剪执行）。它的全年真实费用14,066,257.48元，比同预测exp004无季节高3.3403%，方向反转减少64.6273%、电芯EFC仅减少4.2908%。

已测 `greedy_execution` 对照使用同一条件树/动态规划并改回旧因果执行，每天按自己的真实SOC重新规划，真实费用13,534,196.73元，比同预测基线少77,391.11元（0.5686%）。该对照保留原消融角色；没有按全年测试成绩替换正式组或验证跨种子泛化。两份工作簿分别交付，便于按费用优先目标检查。

正式15项计算墙钟202.34秒。所有数组已独立通过费用、能量守恒、SOC、功率、互斥及时间边界核验；不以小样本测速代替全年实测。旧基线同样没有同时充放电，弱惩罚本身不能解释大部分反转差异。

在本独立工作树根目录构建报告，不需重新运行求解：

```sh
/Users/vegetarianwolf/Projects/CUMCM2026/.venv/bin/python -m reports.build_report_exp006 --complete
```

本地预览应提供整个 `app/dist/`（包括content-addressed快照侧文件），离线交付使用`report.html`。网页下载链接依赖同目录工作簿与证据文件，交付目录应一起保留。只有网页内容和全部图表数据嵌入单文件，下载附件不嵌入。

代码分支为`codex/q2-tree-planning`；`a112933`是早期测速提交。正式运行源文件以`evidence/run_manifest.json`的source_sha256为准，最终代码提交以登记记录为准。既有LP工作树和exp001–004历史成绩不由报告构建器改写。
