# 问题一正式结果

正文以 `Xelatex/数模通用模板.tex` 为唯一入口，问题一引入本目录 `tables.tex`，并使用 `q1_day.png` 和 `q1_dispatch.pdf`。

- `revised.json` 与 `revised.csv`：第八次实验已确认的第二层144槽轨迹，来源为 `data/results/exp008/q1/`。
- `baseline.json` 与 `baseline.csv`：用于说明运行约束费用代价的经济基准。
- `summary.json`、`tables.tex` 与调度图：从同一冻结轨迹重新汇总，不重求调度。
- `q1_day.png`：附件1原始负载、光伏与电价图。
- `../code/q1_reproduce.py`：实际费用优先的两层混合整数规划复现程序。

重新核算并出图：从仓库根目录执行 `.venv/bin/python -m experiments.exp009.q12_assets --q1`。
重新求解：`.venv/bin/python Xelatex/code/q1_reproduce.py --quick --out <新结果目录>`；此命令求解两层主方案及经济基准，不覆盖正式冻结轨迹。

原三层的参数扫描、噪声测试及费用让步图已移入 `paper/archive/旧Q1三层诊断/`，不属于当前正式结果。正文参数敏感性使用 `data/results/exp008/robustness/q1/` 的已验收两层模型试验。
