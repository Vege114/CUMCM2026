# 实验报告与可复用模板

[最新报告索引](latest.md) · [首轮正文](experiments/exp001/report.md) · [交互报告，下载后用浏览器打开](experiments/exp001/report.html) · [指定日期完整表格](experiments/exp001/specified_dates.md)

本目录是可以独立浏览的报告包。`main` 与实验分支包含相同的正文、网页、图表、机器可读成绩和模板。交互网页为单文件离线版本，无需训练环境或外部网络；网页源码及完整数据在 `experiments/exp001/app/`。在 GitHub 直接阅读 Markdown；HTML 下载后在浏览器打开。

## 创建下一次报告

```bash
python reports/new_experiment.py --id exp002 --title "下一次实验的具体名称"
```

这会创建八部分正文草稿和 `record.draft.json`，不会产生虚构成绩或进入正式比较。按 `templates/experiment.schema.json` 填入真实数据校验值、代码提交、环境、种子、模型配置、技术路径、指标定义、预测与调度成绩。技术讲解可参考 `templates/methods-neural-v1.md`，应按下一次真实实现改写。

完成报告与核验后登记：

```bash
python reports/register_experiment.py --record reports/experiments/exp002/record.draft.json
```

登记会读取此前全部记录，并为每个可匹配的调度和预测指标计算“（本次值－此前值）／此前值绝对值”。数据、评价时期、时间对齐、计费、储能物理条件或指标定义改变时，仍保留两次结果和差异说明，但不计算直接排名。零分母标为缺失。模型结构和随机种子允许改变，它们正是技术对比的内容。

`registry/` 是正式实验注册表；每个实验目录保留当时的历史比较快照。通用登记程序拒绝覆盖旧编号，最新索引和相对比较 CSV 则追加更新。每个新交互报告应把同一份历史比较数据接入图表；本轮源码里的 `history` 和 `relative_history` 查询展示了接口。

## 复现首轮报告

实验分支中的 `reports/build_report.py` 从 `data/results/exp001` 读取经过核验的结果，生成正文、静态图、记录和网页数据；它针对本轮固定的三网络、两个特征消融、三个种子协议。通用模板与登记工具不依赖这个训练配置。

网页采用 Data 插件报告组件。修改网页源码时遵守应用目录中的 `AGENTS.md`，只修改内容和数据。使用插件的 `data-app.mjs build --separate-data` 构建，再用 `export-offline` 导出单文件；构建与实际浏览器检查是不同步骤。已经导出的 `report.html` 可直接使用，不依赖该插件。

论文静态图同时保留 PNG 与 SVG。CSV 保留未四舍五入的核算值，正文和网页仅在显示时格式化。当前没有历史训练成绩，首轮如实标为首次实验，简单历史预测只是本轮对照。
