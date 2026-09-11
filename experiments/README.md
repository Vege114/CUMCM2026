# 实验记录

当前可运行版本位于 [baseline_v2/](baseline_v2/README.md)，基础版本保留在 [baseline_v1/](baseline_v1/README.md)。各版代码、参数、已核验的脱敏运行日志、图表和报告集中管理，每局输出保存至对应版本的 `runs/<run_id>/`，不会覆盖既有证据。

[v1 运行报告](baseline_v1/REPORT.md) 包含 Q3/Q4 两场官方演练和六场离线结果；[v2 改进与验证报告](baseline_v2/REPORT.md) 包含另外两场官方演练及 30 对自建场景验证。两份报告均按模块说明实际算法、输入输出、参数、决策依据与流程关系。没有使用正式测试机会。

后续实验使用 [实验记录与报告模板](record-template.md)，其中“模块化技术细节与算法实现”为必填部分。技术说明须对应实际运行代码，明确模块与函数、算法过程、目标函数或判据、保证条件与启发式边界，并提供模块关系图；优化建议单列。更新报告时直接修改主报告原文件，并同步其生成模板或内容来源。v1 的生成模板为 [report_template.md](baseline_v1/report_template.md)；v2 主报告独立维护，`make_report.py` 只生成配对套件的 `COMPARISON.md`。

[v1 比较索引](baseline_v1/benchmark_index.json) 与 [v2 配对结果](baseline_v2/runs/validation_20260911/COMPARISON.md) 可供后续版本读取。

其他尚未核验或私有的运行输出仍可放在被忽略的 `experiments/runs/`。正式导出日志归档至 `submission/logs/problem3/` 或 `problem4/`，保持原名及内容。
