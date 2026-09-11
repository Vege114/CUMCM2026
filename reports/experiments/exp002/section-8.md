## 8. 复现说明

统一入口：`.venv/bin/python -m experiments.common.neural_v2.run STAGE`。STAGE 依次为 prepare、train、predict、calibrate、replay、export、report；也支持 all。训练强制 GPU，各阶段可独立恢复，配置或源代码签名变化会拒绝复用旧缓存。详细命令、导出依赖及缓存约定见 experiments/common/neural_v2/README.md。

正式结果为 result2.xlsx、result3.xlsx、result4-2.xlsx、result4-3.xlsx。购电表的全天总费用已包含原计划、最终净调整和紧急购电，多个工作表中的重复总费用不能相加。逐日 CSV、指定日期表格、论文 PNG/SVG、四个工作簿和离线 report.html 同包交付；体积较大的逐区间 NPZ 留在实验分支的 data/results/exp002。主分支报告和全部网页筛选可以独立使用。

复现边界：有上限的混合整数求解会受到机器速度影响；固定种子与数据并不能保证每次在同一秒数预算下找到同一可行解。已有缓存可精确恢复已完成回放，从头重算需重新报告求解状态与耗时。所有最终核验结果保存为 verification.json、unit_tests.txt、browser_checks.json 和 consistency_checks.json。

本次冻结实验代码提交：`ab359643a10d584546f3c397d021f3ae44487a49`。预测检查点、上游档案、数据与源代码的 SHA-256 见 prediction_archive.json、data_hashes.json、source_hashes.json。
