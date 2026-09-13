# exp005：保持原模型的 D4-A 失败证据

本包按用户最新决定“保持原模型，仅整理失败证据”交付。β=0、D5-A、1000 kW硬爬坡及两层目标保持不变，没有运行β=0.001检验。

2025-02-01 13:30滚动求解的第二层，在未来14:10—14:20返回充电740.656357119186 kWh、放电566.5937811694077 kWh。停止前81段实际动作与当次当前节点均未出现超过容差的重叠。D4-A检查全体规划节点，因此在执行13:30动作前停止认证。尚未证明所有第二层最优解都必须重叠。

## 阅读顺序

1. `report.html`：可离线打开的交互报告，包含两层与窗口范围筛选。
2. `report.md`：按仓库八节报告模板撰写的正文及静态图。
3. `evidence/solver_stages.csv`、`failed_window.csv`：两层结果和失败窗口63段明细。
4. `evidence/executed_prefix.csv`：停止前81段实际轨迹。
5. `evidence/independent_audit.json`、`tests.txt`：独立残差核验及11项测试记录。
6. `evidence/source_hashes.json`：输入和实现文件的SHA-256。

`figures`同时提供PNG和SVG。证据包另含原始运行目录`data/results/exp005/beta0-boundary`，其中`execution-081.npz`为首次失败窗口的求解档案；`failed_window_both_layers.npz`保留相同输入复算的两层完整数组，复算与首次记录最大差为0。

## 复现边界

压缩包中`experiments/problem2/stochastic_lp`、`tests/test_q2_stochastic_lp.py`与`reports/build_report_exp005.py`是本轮源代码快照。它们依赖本项目既有数据、exp003数据加载器、linear_planning基础组件和冻结预测，需在CUMCM2026仓库中运行；压缩包不是独立训练环境。具体命令见报告第八节。

正式334日评价未完成，本包不含`result2.xlsx`、全年费用、节约率或指定四日策略。报告整理完成不代表购电策略通过物理认证。
