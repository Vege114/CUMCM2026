"""Reproduce section 9 figures from archived exp008 evidence; no model reruns."""
from pathlib import Path
import csv
import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
EVIDENCE = ROOT / "reports/experiments/exp008/robustness/evidence"
font_manager.fontManager.addfont(str(ROOT / "Xelatex/fonts/SourceHanSerifCN-Regular.otf"))
FONT = font_manager.FontProperties(fname=str(ROOT / "Xelatex/fonts/SourceHanSerifCN-Regular.otf")).get_name()
plt.rcParams.update({"font.family": FONT, "font.size": 10, "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "axes.labelcolor": "#303030",
                     "text.color": "#303030", "axes.edgecolor": "#777777"})
BLUE = "#356A9A"

def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def save(fig, name):
    for ext, dpi in [("pdf", 300), ("png", 200)]:
        fig.savefig(OUT / f"{name}.{ext}", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

q1path = EVIDENCE / "q1/sensitivity.csv"
noise_path = EVIDENCE / "execution/noise_summary.csv"
data = read_csv(q1path)
assert len(data) == 15 and all(r["outcome"] == "optimal_and_physical_audit_passed" for r in data)
labels = ["费用让步 0.05%", "费用让步 0.09%", "费用让步 0.11%", "费用让步 0.20%",
          "爬坡上限 900 kW", "爬坡上限 1100 kW", "最短开启 2槽", "最短开启 4槽",
          "最短关闭 1槽", "最短关闭 3槽", "往返效率 85.5%", "往返效率 94.5%",
          "容量 90%", "容量 110%"]
rows = [r for r in data if r["name"] != "center"]
base = next(r for r in data if r["name"] == "center")
fig, axes = plt.subplots(1, 2, figsize=(9.2, 6.5), sharey=True)
y = np.arange(len(rows))
for ax, metric, title in zip(axes, ["cost", "tv"], ["(a) 购电费用变化", "(b) 循环功率总变差变化"]):
    values = np.array([(float(r[metric]) / float(base[metric]) - 1) * 100 for r in rows])
    ax.barh(y, values, height=.56, color=BLUE, edgecolor=BLUE, linewidth=.6)
    for j, v in enumerate(values):
        if v < 0:
            ax.patches[j].set_facecolor("white")
            ax.patches[j].set_hatch("///")
        ax.annotate(f"{v:+.3f}%" if abs(v) >= .0005 else "0.000%",
                    (v, j), xytext=(4 if v >= 0 else -4, 0),
                    textcoords="offset points", ha="left" if v >= 0 else "right", va="center", fontsize=8)
    ax.axvline(0, color="#444444", linewidth=.8)
    ax.set_xlim(min(values.min()*1.9, -0.1), values.max()*1.45)
    ax.set_title(title, loc="left", fontsize=11, pad=13)
    ax.set_xlabel("相对基准的变化 / %")
    ax.grid(axis="x", color="#dddddd", linewidth=.5)
    ax.set_axisbelow(True)
    for boundary in [3.5, 5.5, 7.5, 9.5, 11.5]:
        ax.axhline(boundary, color="#eeeeee", linewidth=.7)
axes[0].set_yticks(y, labels)
axes[0].invert_yaxis()
fig.suptitle("问题一单参数重求解的敏感性", fontsize=15, x=.02, ha="left", y=.995)
fig.text(.02, .94, f"15组（含基准）｜基准费用 {float(base['cost']):.2f} 元；循环总变差 {float(base['tv']):.2f} kW", fontsize=10)
fig.text(.02, .005, "来源：exp008 / q1 / sensitivity.csv。各组重新优化；非固定计划扰动检验。\n实心表示增加，斜线表示减少；两个面板的横轴尺度不同。", fontsize=9)
fig.subplots_adjust(left=.21, right=.98, bottom=.14, top=.86, wspace=.22)
save(fig, "q1_sensitivity")

noise = read_csv(noise_path)
assert len(noise) == 4 and all(int(r["replicates"]) == 30 for r in noise)
fig, ax = plt.subplots(figsize=(8.2, 4.1))
for j, r in enumerate(noise):
    mean, lo, hi = (float(r[k]) for k in ["mean_cost_change_pct", "p05_cost_change_pct", "p95_cost_change_pct"])
    ax.errorbar(mean, j, xerr=[[mean-lo], [hi-mean]], fmt="o", color=BLUE,
                elinewidth=2, capsize=5, markersize=6)
    ax.text(hi+.7, j, f"{mean:.3f}%  [{lo:.3f}, {hi:.3f}]", va="center", fontsize=10)
ax.set_yticks(range(4), [f"σ = {float(r['log_noise_std']):.2f}" for r in noise])
ax.invert_yaxis()
ax.set_ylim(3.35, -.35)
ax.set_xlim(0, 60)
ax.set_xlabel("相对未扰动历史计划的购电费用变化 / %")
ax.set_ylabel("对数噪声标准差")
ax.grid(axis="x", color="#dddddd", linewidth=.5)
ax.set_axisbelow(True)
ax.set_title("第二问固定计划的相关噪声执行回放", loc="left", fontsize=14, pad=35)
ax.text(0, 1.04, "2025年2—12月，334日｜每档30次｜圆点为均值，横线为5%—95%经验分位", transform=ax.transAxes, fontsize=9)
fig.text(.02, .015, "来源：exp008 / execution / noise_summary.csv。\n固定购电与模式，仅扰动执行供需；经验分位不是未来费用的置信区间。", fontsize=9)
fig.subplots_adjust(left=.15, right=.98, top=.76, bottom=.23)
save(fig, "q2_execution_stress")

manifest = {
    "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [q1path, noise_path, EVIDENCE / "execution/protocol.json"]},
    "q1": {"rows": 15, "comparison": "each scenario reoptimized; percentage change relative to center",
           "chart": "two aligned horizontal diverging bar panels; all 14 nonbaseline scenarios"},
    "q2": {"rows": 4, "replicates_each": 30, "chart": "mean with empirical p05-p95 interval",
           "scope": "archived exp008 fixed-plan execution replay, not exp009 retraining"},
    "surface": "standalone PNG previews and vector PDF for LaTeX",
    "palette": {"policy": "single-root preferred", "blue": BLUE, "negative": "open fill with hatch"},
    "checks": "15 valid q1 rows, 4 noise levels with 30 repeats each; no model recomputation",
}
(OUT / "sources.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print("Generated two PDF/PNG figures and sources.json")
