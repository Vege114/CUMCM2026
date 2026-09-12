# 实验报告与可复用模板

[exp004 预测与季节性报告](experiments/exp004/report.md) · [exp004 离线交互报告](experiments/exp004/report.html) · [预测交接接口](experiments/exp004/README-prediction.md)

exp004在`codex/q2-discussion-seasonality`分支完成，只修改第二问预测；规划与储能执行沿用exp003。比较无季节、历史季节和含未来信息的全年探索，报告同时保留exp001–003的绝对费用、电量、预测误差与相对变化。三份题目工作簿分别位于该报告的三个组目录。

[最新报告索引](latest.md) · [exp003 第二问报告](experiments/exp003/report.md) · [exp003 离线报告](experiments/exp003/report.html) · [exp002 正文](experiments/exp002/report.md) · [exp002 离线交互报告](experiments/exp002/report.html) · [指定日期完整表格](experiments/exp002/specified_dates.md) · [exp001 原报告](experiments/exp001/report.md)

本目录是可以独立浏览的报告包。`main` 与实验分支包含相同的正文、网页、图表、机器可读成绩和模板。交互网页为单文件离线版本，无需训练环境或外部网络；每轮网页源码及完整数据在相应 `experiments/expNNN/app/`。在 GitHub 直接阅读 Markdown；HTML 下载后在浏览器打开。

## 创建下一次报告

```bash
python reports/new_experiment.py --id exp004 --title "下一次实验的具体名称"
```

这会创建八部分正文草稿和 `record.draft.json`，不会产生虚构成绩或进入正式比较。按 `templates/experiment.schema.json` 填入真实数据校验值、代码提交、环境、种子、模型配置、技术路径、指标定义、预测与调度成绩。技术讲解可参考 `templates/methods-neural-v1.md`，应按下一次真实实现改写。

完成报告与核验后登记：

```bash
python reports/register_experiment.py --record reports/experiments/exp004/record.draft.json
```

登记会读取此前全部记录，并为每个可匹配的调度和预测指标计算“（本次值－此前值）／此前值绝对值”。数据、评价时期、时间对齐、计费、储能物理条件或指标定义改变时，仍保留两次结果和差异说明，但不计算直接排名。零分母标为缺失。模型结构和随机种子允许改变，它们正是技术对比的内容。

`registry/` 是正式实验注册表；每个实验目录保留当时的历史比较快照。通用登记程序拒绝覆盖旧编号，最新索引和相对比较 CSV 则追加更新。每个新交互报告应把同一份历史比较数据接入图表；本轮源码里的 `history` 和 `relative_history` 查询展示了接口。

每次与此前实验对比时，必须在对话里通过 Visualize 直接展示比较图，同时保留报告中的对应图表。费用、预测误差和阶段耗时分别呈现，标清原登记或新协议重算、数据单位、改善及退步。详细要求见 [历史比较可视化要求](templates/history-comparison.md)；[相对变化图模板](templates/history-comparison.inline.html) 可接入核验后的比较数据。下一轮报告草稿的第七部分会自动带上这项要求。

## 复现首轮报告

实验分支中的 `reports/build_report.py` 从 `data/results/exp001` 读取经过核验的结果，生成正文、静态图、记录和网页数据；它针对本轮固定的三网络、两个特征消融、三个种子协议。通用模板与登记工具不依赖这个训练配置。

网页采用 Data 插件报告组件。修改网页源码时遵守应用目录中的 `AGENTS.md`，只修改内容和数据。使用插件的 `data-app.mjs build --separate-data` 构建，再用 `export-offline` 导出单文件；构建与实际浏览器检查是不同步骤。已经导出的 `report.html` 可直接使用，不依赖该插件。

论文静态图同时保留 PNG 与 SVG。CSV 保留未四舍五入的核算值，正文和网页仅在显示时格式化。exp001 的首次实验说明保持原貌；新报告自动比较全部此前登记。

## 固定网络与风险调度报告

`build_report_v2.py` 与 `templates/methods-neural-v2.md` 支持固定网络、三个种子独立评价、多阶段场景树、费用尾部风险以及旧正式预测在新物理口径下的重算。八部分结构不变，不要求出现三类候选网络。`v2_evidence.py` 从冻结产物生成样本分解、历史正式预测比较和论文图，不触发训练或调度。

报告构建器直接读取已交付的 `predictions.npz` 和训练元数据，不需要本机的 `.keras` 权重或月度训练缓存。重建报告仍需实验分支中的代码、原始数据和结果目录；在 `main` 独立阅读 HTML、Markdown 和工作簿不需要这些训练依赖。

```bash
python reports/build_report_v2.py --experiment exp002
python reports/verify_delivery_v2.py --experiment exp002
```

构建器默认使用本机 Codex 捆绑 Node 与 Data 插件，也可通过 `--node`、`--data-plugin` 指定安装位置。`--snapshot-only` 只重建正文与完整数据快照；`--partial` 生成明确标示未完成的预览，不得作为正式全年结论交付。

每次报告的机器可读成绩、模板、论文图、工作簿与 HTML 同包保存，网页所有交互数据已完整内嵌。需要从头训练或调度时切换至相应实验分支；在 `main` 阅读报告不依赖训练模块。

## 第二问独立优化报告

exp003 从 v2 分支分出，只改变第二问。八节结构保留，完整方法和候选放在附件；历史比较仅含第二问，旧报告与登记保持不变。`q2_comparison.py` 从同午夜样本与同物理费用生成范围明确的比较，`build_report_q2.py` 构建离线报告。此轮独立校验记录后登记，未使用会追加全题历史表的通用登记程序。复现命令见 [实验说明](../experiments/problem2/exp003/README.md)。
