# 问题二至四精简的独立验收

本目录只记录论文精简、数据保持与实际排版的独立审查。验收者没有修改被审正文、模型、结果表、轨迹或工作簿，没有重新运行训练或优化。

- `W1_review.md`：授权范围、保留目标与技术路线、局部符号和引用的人工审查。
- `verify_source.py` / `source_verification.json`：保护文件、严格范围例外、指定日期数值映射、目标式和引用的独立复验。
- `baseline.json` / `baseline_build.json`：最终公平基线为远程 b24 提交，仅包含用户同样授权的两处问题一引用修复；`baseline_raw_build.json` 保留修复前引用预警。中间 cfea 基线另存，不作为最终净省页起点。
- `verify_pdf.py` / `pdf_verification.json`：PDF 逐页扫描、保留公式与释义首行、保护前缀逐字符位置比较以及页数变化。
- `visual_review.md` / `gate_summary.json`：最终版实际渲染检查和门禁结论。

源码复验命令：

```sh
.venv/bin/python reports/experiments/exp009/q234_compression_audit/verify_source.py
```

PDF 复验需要安装 `pdfplumber` 的 Python、当前编译 PDF，以及 `baseline.json` 指向的隔离编译基线。基线目录属于忽略的构建产物；若在新克隆中复验，需要从记录的 Git 提交恢复 LaTeX 源，仅应用记录的两处引用替换，再按 `baseline_build.json` 的构建记录生成基线。

```sh
python reports/experiments/exp009/q234_compression_audit/verify_pdf.py
```

后一个命令默认检查 `Xelatex/build/数模通用模板.pdf`，可用第一个参数指定另一个相同内容的 PDF。缺少比较基线时会明确失败，不把缺失的保护前缀核验标为通过。实际渲染预览保存在忽略目录 `tmp/conflict-resolution/q234-compression/audit/`。
