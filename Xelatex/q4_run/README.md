# 第四问论文支持材料

正文入口为 `../sections/08-问题四.tex`；独立预览入口为 `../第四问预览.tex`，在 `Xelatex` 目录用 XeLaTeX 编译。主文仍由 `../数模通用模板.tex` 引入第四问。

本目录沿用用户选定的第八次实验结果。`data/dispatch_4-2.npz`、`data/dispatch_4-3.npz` 为334日逐槽轨迹，`price_predictions.npz` 为同一期午夜价格预测，`warmup_*.npz` 为一月历史基准预热。`data/source_manifest.json` 记录来源与哈希。这里没有重新训练预测器或重新求解购电计划。

从仓库根目录运行：

```sh
python Xelatex/q4_run/q4_assets.py
```

依赖：NumPy、pandas、Matplotlib、Pillow。程序验证档案哈希、SOC连续性、能量平衡、功率边界及真实账单，随后重建 `paper/` 中CSV、TeX表和三组PDF/SVG/450 dpi PNG配图。配图使用项目现有YaHei Consolas TrueType字体，正文沿用思源宋体；`vendor/` 保留math-modeling Skill原始配图样式、导出与版面检查脚本。

- `q4_price`：同一48096槽的价格误差；箱体为IQR，须为5%与95%分位，无置信区间含义。
- `q4_monthly`：真实月费用、紧急费占比；不把不同策略差解释为单因素效应。
- `q4_dispatch`：题目指定9月23日的价格、购电、电池功率和储电量；价格预测曲线都是午夜版本。
- `purchase_table.tex`：六个指定购电槽、全天原计划/最终电量与总费用。
- `storage_tables.tex`：六个四小时充放电段及日初、日末SOC。
- `emergency_tables.tex`：四个指定日全部连续紧急购电区间及电量。

`result4-2.xlsx`、`result4-3.xlsx` 从已验收的原成果分支原样保留。4-3的保存轨迹仅上调，故按50%下调退款的账单复算，金额亦不变。与main中其他章节的差异单独见 `../第四问冲突清单.md`，不进入正文。
