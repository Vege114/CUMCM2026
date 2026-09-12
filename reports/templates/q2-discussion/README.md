# 第二问人工讨论模板

`template.html` 是可独立打开的 HTML 空证据模板。没有演示数值，也不会启动训练。它保留第二问题意、模块分工、技术选项和会议记录区；数据与费用图会明确提示尚未绑定证据。

本模板与 exp003 实验报告使用不同会议结构：

1. 从题意和总购电费公式出发，统一边界。
2. 阅读原始电价、负载、光伏。
3. 比较已运行方案的总购电费及计划费、紧急费。
4. 按模块查看输入、输出、交接检查和可选技术。
5. 对照题目表1—表3阅读指定日期。
6. 填写负责人、技术意向及理由，导出会议草稿。

## 可复用文件

- `ReportContent.jsx`：会议型报告内容组件；所有图表使用共享来源绑定。
- `report.css`：仅作用于本报告内容的布局与样式。
- `template.html`：共享运行时导出的独立 HTML；可离线使用。
- `inline-overview.html`：Visualize 的概览片段模板，由相邻构建脚本填入证据。

数据结构与构建入口保存在 `reports/discussions/q2-improvement/`：`build_report.py` 只读取随包 `evidence.json` 和 `modules.json`。它不依赖 exp003 的算法、模型或结果目录，因此在 main 分支也能重建报告。

## 重建与复用

仓库根目录执行：

```sh
python reports/discussions/q2-improvement/build_report.py --complete --template
```

构建需要本地安装的 Data Analytics 共享运行时。默认自动定位当前桌面环境的 Node 与插件；在另一台设备上可显式指定：

```sh
python reports/discussions/q2-improvement/build_report.py \
  --node /absolute/path/to/node \
  --plugin-root /absolute/path/to/data-analytics \
  --complete --template
```

`--template` 同时重建空证据模板，使用独立 artifact ID；报告的现有 ID 会被保留。运行时模板工程保存在仓库 `.work/q2-discussion/template-app/`，不作为报告内容发布。交付的 `template.html` 无需这些构建依赖即可阅读。

复用于下一轮讨论时，先复制整套讨论目录与本模板内容源，并使用官方 preparer 创建新 app。更新 `modules.json` 的模块、技术决定和解释；全体候选必须先写入新的讨论选项日志。将真实数据按 `evidence.json` 的结构绑定后，再生成实测图与结论。修改计算口径需要更新证据，不可只编辑图标题。

## 会议草稿

每个技术决定默认“待讨论，尚未选择”。页面选择只保存在当前浏览器、当前 artifact ID 对应的本地状态。导出 JSON 包含负责人、选项、理由及候选目录；`model_changed` 固定为 `false`。草稿不是模型配置，也不是已经批准执行的新实验。

空模板中的既有做法用于说明第二问当前求解结构，图表证据保持为空；空白不等于零，不可视为实测绩效。
