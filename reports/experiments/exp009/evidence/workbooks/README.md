# exp009 工作簿核验

五份正式工作簿为报告目录的 `result1.xlsx`、`result2.xlsx`、`result3.xlsx`、`result4-2.xlsx`、`result4-3.xlsx`，共 16 张工作表，沿用 `data/templates/` 的原附件模板。`outputs/exp009-cost-only-control/` 仅保留同字节副本，不是额外候选。

`export_workbooks.mjs` 使用 bundled Node 的 `@oai/artifact-tool`，直接填入已通过独立验证的 exp009 载荷。问题一明确验证只有费用阶段1，没有继承 exp008 的阶段2或旧8%停止门槛。四个年度场景覆盖 334 日、每天144时段，充放电表按原模板汇总为六个四小时块。

调整购电表填写最终实际采用的绝对购电量；它不表示相对原计划的差量。两个购电表中的每日总费用是同一笔结算账单，仅计一次。购电、充电、放电和储电量均为 kWh，费用为元。第一问原轨迹中的充放电功率在载荷中仅除以6一次，导出时不再次转换。

优化结果作为数值保存。年度购电量总计使用 `SUM(B行:EO行)`，并在保存后验证第一日、中间日和最后一日的输入改动会更新总计，恢复输入后总计也恢复。

- `independent_cell_audit.json`：独立 Python XML 读取器直接从 NPZ/问题一 JSON 重建输出，逐项比较 350,931 个数据/表头单元格；不调用报告载荷构建器、优化器或导出器。
- `saved_workbook_audit.json`：Artifact Tool 从已保存文件重导入，比较 352,935 项（含 2,004 条求和公式），执行18组求和输入改动及恢复检查，扫描公式错误并渲染各表。
- `visual_review.json`：16张工作表的视觉覆盖、首尾日和时段核对、单位和总量表头检查，以及最终文件哈希。视觉修订仅将年度购电表 EP/EQ 两列扩至160像素，修复单位括号单独换行，数值及公式保持相同。
- `result*-export.json`：原模板、源轨迹、载荷、输出文件及相同副本哈希。

渲染和公式检查使用 Artifact Tool，未启动 Excel 桌面应用，也未声称 Excel 可重算外部优化模型。四小时块内两种电量都为正可来自不同十分钟时段，不能将块汇总解释为同时充放电。

运行入口为 `experiments/exp009/export_workbooks.mjs`、`verify_workbooks.mjs` 和 `verify_workbook_sources.py`。所有 XLSX 均由 Artifact Tool 创建；Python 仅用于独立读取和数值审计。正式创建前已唯一执行一次 `mark_artifact_operation_started.mjs --operation-kind create --expected-output-count 5 --output-format xlsx`。
