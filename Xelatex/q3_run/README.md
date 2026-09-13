# 第三问：退款口径与严格物理情景规划

本目录随附计算输入、完整求解与检验代码、冻结结果和论文图表。输入路径相对于脚本所在目录，不依赖原仓库、用户上传目录或个人绝对路径。

## 直接运行

在有 Python 3.12 或兼容版本的环境中执行：

```bash
python -m pip install -r requirements.txt
python q3_reproduce.py --case main
python q3_reproduce.py --case no_update
python q3_analysis.py
python q3_tables.py
```

已附带完整结果时，求解器拒绝覆盖成功档案。若要重新运行整套流程，请先复制本目录并将副本中旧的 `results` 改名保留；分析程序只接收完整365日回放，正式统计最后334日。也可用 `--case my_run` 输出新的单个策略目录；名字 `no_update` 专指取消日内更新的对照。

HGB午夜预测来自既有 exp008 的单HGB、Ridge28和日偏差记忆管线，作为本问输入，包含完整发布日期。程序不是HGB重新训练程序。官方预报节点校准、日内负载修正、历史误差场景、MILP、实际回放和检验均在随包代码中实现。原始CSV读取兼容UTF-8与GB18030，缺失或损坏时明确失败，不能回退到未来实测数据或虚构数据。

`q3_reproduce.py` 内有物理、计费与求解可行性检查；`q3_reproduce_annotated.py` 是增加逐行中文注释的完整可运行阅读版，已通过AST等价检查，运算逻辑与正式程序相同；`q3_analysis.py` 包括未来数据扰动、退款算例、四季指定日风险参数±10%敏感性、预测评分和图表；`q3_tables.py` 从相同结果生成正文表格与工作簿载荷。

## 输出

- `results/main/dispatch_all.npz`：新退款模型365日真实轨迹，一月仅预热。
- `results/no_update/dispatch_all.npz`：相同输入层下仅午夜计划的完整策略对照。
- `results/*/protocol.json`：源程序及输入SHA-256、求解配置。
- `results/*/solver_audit.json`：每次发布的场景历史、可行性、间隙与耗时。
- `paper/all_intervals.csv`：完整334日十分钟购电、充放电、紧急购电与SOC。
- `paper/specified.csv`、`storage.csv`、`emergency.csv`：四个题目指定日结果。
- `paper/analysis.json`：检验、预测指标、参数敏感性与策略对照。
- `paper/q3_results.tex`、`q3_validation.tex`：由同一轨迹生成的正文结果与检验表。
- `paper/q3_*.png`：300 dpi论文图。
- `result3.xlsx`：附件格式工作簿，原计划与最终常规购电分别保留，修正原模板十分钟时间标签。

工作簿导出与Python数值求解分离。`export_workbook.mjs` 使用 Node.js 和 `@oai/artifact-tool`，读取 `paper/workbook_payload.json` 填写随包模板；具备该库的环境执行 `node export_workbook.mjs`。普通Python环境可直接复现完整数值、CSV和论文图表，无需该库。

## 解释边界

真实费用为原计划费＋1.5倍上调费－0.5倍下调退款＋5倍紧急费，按最终相对原计划的净偏差一次结算。规划场景和实际执行均禁止同时充放电及紧急电充电。不额外加入1000kW爬坡或最短启停硬限制。

场景补救为开环近似；未来更新的1.5倍短缺权重为代理，不是完整多阶段最优模型。有限秒数下的MILP间隙针对当次代理目标，不能解释成实际年度费用的最优性误差。不同机器或负载下限时可行解可能不同。正式年度已经参与开发，不能称未触碰测试集。

两组从一月初相同6000kWh独立运行，二月初库存可能不同；正文披露各自初态，年度费用差代表完整策略对照。四季参数实验采用相同主方案日初状态，仅是指定日局部敏感性，不冒充完整年度鲁棒性结论。
