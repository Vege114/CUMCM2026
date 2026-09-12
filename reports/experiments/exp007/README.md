# exp007 报告包

六组 PPO 规划正式实验已完成，计算墙钟 574.90 秒。预声明主组固定 `regularized_42`，未按最终费用挑选种子。报告模板内容来自 main `7ecd9b9`，最终复查远程 main `7fffa31` 后确认模板未变。

主组总费用 **17,925,755.58 元**，比 exp006 正式树 DP 高 **27.44%**；电池等效全循环 **247.42 次**，减少 **48.21%**；非空方向反转 **1,129 次**，减少 **58.63%**。六组均无同时充放电，且费用均高于 exp006。电池运行强度下降不能解释为已验证的寿命收益。

## 打开结果

- `report.html`：内嵌完整交互数据的离线报告；请保留整个报告目录，以便下载工作簿和证据附件。
- `report.md`、`methods.md`：正文与技术方法。
- `result2.xlsx`：预声明主组题目结果；`cost_only_42/result2.xlsx`：纯费用对照。
- `evidence/report_data.json`：全部显示数据；其余 CSV/JSON 记录费用、误差、耗时、电池强度、相对变化及检查结果。
- `figures/`：31 组 PNG/SVG，包括九种策略每种 48,096 点的完整电池曲线。
- `specified_dates.md`、`specified_tables/`：题目指定日期表格。
- 仓库根目录的 `data/results/exp007/`：保存六组原始执行、训练日志与 144 个检查点。

历史比较覆盖 exp001–006。exp005 虽使用同源输入，但额外硬爬坡、紧急充电规则不同，只作描述比较；全年信息预言机、旧物理口径和探索组均有明确标记，不混入同口径排名。预测沿用 exp004 `no_season/seed42`，RL 的三个种子只改变规划训练。

## 复核和重建

以下命令从本分支仓库根目录执行。此机器复用项目环境；其他机器可将 `EXP007_PY`、`EXP007_NODE` 替换为安装了对应依赖的解释器。原始环境见 `data/results/exp007/pilot/environment.json`，Python 依赖见仓库配置，报告 app 的 npm 依赖见 `app/package-lock.json`。

```sh
EXP007_PY=/Users/vegetarianwolf/Projects/CUMCM2026/.venv/bin/python
EXP007_NODE=/Users/vegetarianwolf/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/private/tmp/exp007-matplotlib XDG_CACHE_HOME=/private/tmp/exp007-cache

"$EXP007_PY" -m unittest discover -s tests -p 'test_q2_rl_*.py' -v
"$EXP007_PY" -m experiments.problem2.rl_planning.verify
"$EXP007_PY" -m experiments.problem2.rl_planning.export --node "$EXP007_NODE"
"$EXP007_PY" -m experiments.problem2.rl_planning.export --node "$EXP007_NODE" --case cost_only_42
"$EXP007_PY" -m reports.exp007_evidence
"$EXP007_PY" -m reports.exp007_figures
"$EXP007_PY" -m reports.build_report_exp007 --complete --node "$EXP007_NODE"
"$EXP007_PY" -m reports.register_exp007
```

工作簿导出使用 Codex 的 artifact-tool 运行时，报告构建使用本机 Data 插件工具。跨机器时可通过 builder 的 `--data-plugin` 参数指定 Data 插件路径。重新导出会改变生成时间及对应文件哈希，需要重新检查并更新最终交付索引；不会重新训练。

正式入口为 `"$EXP007_PY" -m experiments.problem2.rl_planning.run`。当前完整归档会使入口直接退出。完整重训请另建隔离副本和空输出目录，使用执行提交 `d5ed43b` 的冻结源码、批准协议与同一输入；不要删除或覆盖当前证据。原始测速不是预训练结果，也不构成正式成绩。

## 核验范围

`evidence/independent_qa.json` 核验六组的固定训练预算、因果截止、源码与输入签名、每日锁定计划、跨日状态和真实账单；`evidence/tests_final.txt` 记录 15 项测试结果。两份工作簿的 QA、静态图检查、完整曲线导出核验和浏览器最终检查分别归档在各自 evidence 文件中。原始六组计算连续完成，未进行完整进程崩溃注入试验。
