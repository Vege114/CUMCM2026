# 最终成果工作簿核验

五份成果位于 `../../result1.xlsx`、`../../result2.xlsx`、`../../result3.xlsx`、`../../result4-2.xlsx`、`../../result4-3.xlsx`。按题目 Excel 模板保留工作表与列结构，填充冻结最终载荷。数据副本同时保存在 `data/results/exp008/`，与报告目录交付文件逐字节一致。

`saved_workbook_audit.json` 记录已保存 XLSX 的 SHA-256、对应源档案、逐表核验范围、公式和预览。五份文件共有 16 张工作表，已全部渲染并视觉检查。`detail_previews.json` 另列年度表期末七日、求和及费用列、日内更新区间预览。文字、日期、四位小数电量、两位小数费用和区间终点显示完整。源模板和重建后的全部数据不混用。

全量回读比较 353,463 个数值、标签或公式单元，334 日日期连续、每日 144 个十分钟购电量、每日 6 个四小时电池电量块及全部紧急购电事件均匹配载荷。第一问有 144 个购电量与 6 个电池电量块。年度每天的 `EP` 求和公式逐行核验，并在每张购电表首、中、末行分别临时改变输入再恢复，18 次重算检查全部通过。五份公式错误扫描均为 0。

`source_to_payload_audit.json` 不依赖导出器，直接重算 NPZ 到载荷的计划、最终购电量、四小时电量、期初期末 SOC、连续紧急事件和日结算费用。第一问的 `g` 原本就是 kWh，充、放电功率 `c/d` 才需除以 6 转成十分钟电量；已逐值核验。所有成果工作簿的购电、充放电、储能列单位均为 kWh，费用为元。四小时块内充、放电总量均为正，并不表示十分钟内同时充放。

第三问和第四问更新场景的“调整购电量”保存实施的最终购电量，而非增减量；两张购电表的 `EQ` 列均重复展示同一日最终结算费，汇总时只能计一次。无紧急事件日保留“无”和零电量，缺失日期没有补造。

公式计算、导出和回读使用技能指定的 bundled Artifact Tool。没有启动 Excel 桌面程序，未声称验证其原生运行环境。

复现入口：先由 `final_payload.json` 的各场景 `workbook_data` 写入 `.work/exp008/workbook-{scenario}.json`，再使用 bundled Node 依次执行：

```text
reports/export_workbooks_v2.mjs ROOT exp008 export
experiments/exp008/export_q1_workbook.mjs ROOT --export ROOT/reports/experiments/exp008/workbook-previews/q1 ROOT/reports/experiments/exp008/final_payload.json
experiments/exp008/verify_final_workbooks.mjs ROOT ROOT/reports/experiments/exp008/final_payload.json
experiments/exp008/render_final_workbook_details.mjs ROOT
```

独立原始数组核验使用 bundled Python 执行 `-m experiments.exp008.final_payload_source_audit`。本次运行通过 `.work/exp008/` 中指向源脚本的软链接与指向 bundled `node_modules` 的软链接解析依赖，Node 带 `--preserve-symlinks-main`；未修改旧版导出器。
