"""Build publication-style figures and a Chinese reader report from verified logs."""
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT.parents[1]/".work"/"matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

COLORS = ["#237A9B", "#E89B35", "#5AAE82", "#9B78BE", "#D66B74"]
plt.rcParams.update({"font.family": ["Microsoft YaHei", "DejaVu Sans"], "font.size": 10,
                     "axes.unicode_minus": False, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 140,
                     "savefig.facecolor": "white", "axes.titlepad": 13})
plt.rcParams["svg.hashsalt"] = "jammers-baseline-v1"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def save(fig, name):
    fig.savefig(ROOT/"figures"/f"{name}.png", bbox_inches="tight")
    svg_path = ROOT/"figures"/f"{name}.svg"
    fig.savefig(svg_path, bbox_inches="tight", metadata={"Date": None})
    # Matplotlib paths contain trailing spaces; normalize for clean source diffs.
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines())+"\n",
                        encoding="utf-8", newline="\n")
    plt.close(fig)


def draw_metrics(runs):
    labels = [f"问题{r['metadata']['problem']}\n单次官方演练" for r in runs]
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.8), layout="constrained")
    specs = [("cleared_ratio", "清除比例", "比例（%）", 100),
             ("average_localization_clear_time_s", "平均定位清除时间", "虚拟秒 / 个", 1),
             ("program_runtime_s", "程序运行时间近似值", "现实秒（响应时间戳差）", 1)]
    for ax, (key, title, ylabel, scale) in zip(axes, specs):
        values = [r["official_metrics"][key]*scale for r in runs]
        bars = ax.bar(labels, values, color=COLORS[:2], width=0.48)
        ax.set(title=title, ylabel=ylabel, ylim=(0, max(values)*1.24))
        if key == "cleared_ratio":
            ax.set_yticks(range(0, 101, 20))
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
        for bar, value, run in zip(bars, values, runs):
            text = (f"100%\n{run['official_metrics']['cleared_count']}/{run['official_metrics']['total_jammers']}"
                    if key == "cleared_ratio" else f"{value:.3f}" if key == "program_runtime_s" else f"{value:.2f}")
            ax.text(bar.get_x()+bar.get_width()/2, value+max(values)*0.025, text, ha="center", va="bottom")
    save(fig, "official_metrics")


def draw_breakdown(runs):
    keys = ["movement", "channel_switch", "measurement", "optical_localization", "laser_clearance"]
    names = ["移动", "频道切换", "检测", "光学定位", "清除"]
    fig, ax = plt.subplots(figsize=(11.5, 3.8), layout="constrained")
    lefts = [0.0]*len(runs)
    for key, name, color in zip(keys, names, COLORS):
        values = [r["time_breakdown_s"][key]/60 for r in runs]
        ax.barh(range(len(runs)), values, left=lefts, color=color, label=name, height=0.42)
        if key == "movement":
            for i, value in enumerate(values):
                percent = value*60/runs[i]["official_metrics"]["virtual_total_time_s"]*100
                ax.text(value/2, i, f"移动 {value:.1f} 分钟（{percent:.1f}%）", ha="center", va="center", color="white")
        lefts = [a+b for a, b in zip(lefts, values)]
    for i, value in enumerate(lefts):
        ax.text(value+3, i, f"总计 {value:.2f}", va="center")
    ax.set(yticks=range(len(runs)), yticklabels=[f"问题{r['metadata']['problem']}" for r in runs],
           xlabel="虚拟时间（分钟）", title="整场耗时分解：包含最后一次清除之后的排查", xlim=(0, max(lefts)*1.22))
    ax.invert_yaxis()
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False)
    save(fig, "time_breakdown")


def draw_timeline(runs):
    fig, ax = plt.subplots(figsize=(11.5, 4.3), layout="constrained")
    for run, color in zip(runs, COLORS):
        timeline = run["clearance_timeline"]
        total = run["official_metrics"]["total_jammers"]
        xs = [0]+[t["virtual_time_s"]/60 for t in timeline]+[run["official_metrics"]["virtual_total_time_s"]/60]
        ys = [0]+[t["cleared_count"]/total*100 for t in timeline]+[run["official_metrics"]["cleared_ratio"]*100]
        ax.step(xs, ys, where="post", color=color, linewidth=2,
                label=f"问题{run['metadata']['problem']}（{total} 个）")
        ax.plot(xs[-2], ys[-2], "o", color=color, markersize=5)
    ax.set(title="累计清除进度（单次演练；不同题型、不同随机案例）", xlabel="虚拟时间（分钟）",
           ylabel="累计清除比例（%）", ylim=(0, 108))
    ax.grid(alpha=0.18)
    ax.legend(loc="lower right", frameon=False)
    save(fig, "clearance_timeline")


def draw_routes(runs):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6), layout="constrained")
    for ax, run, color in zip(axes, runs, COLORS):
        folder = ROOT/"runs"/run["metadata"]["run_id"]
        events = read_jsonl(folder/"events.jsonl")
        path = [(0, 0)]
        clears = []
        for event in events:
            req, resp = event["request"], event["response"] or {}
            if resp.get("accepted") and "position" in req:
                p = (req["position"]["x"], req["position"]["y"])
                if p != path[-1]:
                    path.append(p)
                if resp.get("clear_result") == "success":
                    clears.append((p, req["channel"]))
        xs, ys = zip(*path)
        ax.plot(xs, ys, color=color, alpha=0.7, linewidth=1)
        ax.add_patch(Circle((0, 0), 1800, fill=False, linestyle="--", color="#7B8690", linewidth=1))
        ax.scatter([0], [0], marker="s", color="#202A32", s=35, label="起点")
        for p, channel in clears:
            ax.scatter(*p, marker="*", s=55, color="#BB5055", zorder=4)
            ax.annotate(str(channel), p, xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.scatter([], [], marker="*", color="#BB5055", s=55, label="成功清除点（标号为频道）")
        ax.set(title=f"问题{run['metadata']['problem']}：{run['diagnostics']['travel_distance_m']/1000:.2f} km",
               xlabel="东向坐标 x（米）", ylabel="北向坐标 y（米）", xlim=(-2450, 2450), ylim=(-2450, 2450), aspect="equal")
        ax.grid(alpha=0.15)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False, fontsize=9)
    save(fig, "routes")


def diagnostic_details(run):
    decisions = read_jsonl(ROOT/"runs"/run["metadata"]["run_id"]/"decisions.jsonl")
    fallbacks = [d for d in decisions if d["event"] == "optical_fallback"]
    return {"optical_fallback_count": len(fallbacks), "optical_fallback_channels": [d["channel"] for d in fallbacks],
            "after_last_clear_s": run["official_metrics"]["virtual_total_time_s"]-run["clearance_timeline"][-1]["virtual_time_s"],
            "clear_reasons": {reason: sum(d["event"] == "clear_attempt" and d["reason"] == reason and d["success"] for d in decisions)
                              for reason in ("guaranteed_enclosing_circle", "opportunistic_estimate", "guaranteed_grid_cover", "near")}}


def build_report(runs, all_runs):
    q3, q4 = runs
    details = [diagnostic_details(r) for r in runs]
    table = []
    for run in runs:
        m = run["official_metrics"]
        table.append(f"| 问题{run['metadata']['problem']} | `{m['case_code']}` | {m['cleared_count']}/{m['total_jammers']} | {m['cleared_ratio']:.0%} | {m['virtual_total_time_s']:.6f} | {m['average_localization_clear_time_s']:.2f} | {m['program_runtime_s']:.3f} |")
    diag_rows = []
    for title, key, fmt in [("移动路程（m）", "travel_distance_m", ".2f"), ("检测次数", "measure_count", "d"),
                            ("频道切换次数", "channel_switch_count", "d"), ("无信号比例", "no_signal_ratio", ".2%"),
                            ("清除尝试次数", "clear_attempt_count", "d"), ("清除失败次数", "clear_failure_count", "d"),
                            ("拒绝请求数", "rejected_request_count", "d"), ("通信错误数", "transport_error_count", "d")]:
        diag_rows.append(f"| {title} | {format(q3['diagnostics'][key],fmt)} | {format(q4['diagnostics'][key],fmt)} |")
    offline = []
    for problem in (3, 4):
        group = [r for r in all_runs if r["metadata"]["source"] == "local_mock" and r["metadata"]["problem"] == problem]
        avgs = [r["official_metrics"]["average_localization_clear_time_s"] for r in group]
        offline.append(f"| 问题{problem} | {len(group)} | {sum(r['validation']['all_cleared_verified'] for r in group)}/{len(group)} | {statistics.mean(avgs):.2f} | {statistics.median(avgs):.2f} | {max(avgs):.2f} |")
    text = (ROOT/"report_template.md").read_text(encoding="utf-8")
    replacements = {"OFFICIAL_TABLE": "\n".join(table), "DIAGNOSTIC_TABLE": "\n".join(diag_rows),
                    "OFFLINE_TABLE": "\n".join(offline), "Q3_TAIL": f"{details[0]['after_last_clear_s']:.2f}",
                    "Q3_MOVE_SHARE": f"{q3['time_breakdown_s']['movement']/q3['official_metrics']['virtual_total_time_s']:.2%}",
                    "Q4_MOVE_SHARE": f"{q4['time_breakdown_s']['movement']/q4['official_metrics']['virtual_total_time_s']:.2%}",
                    "Q4_FALLBACKS": str(details[1]["optical_fallback_count"]),
                    "Q4_FALLBACK_CHANNELS": str(details[1]["optical_fallback_channels"]),
                    "Q3_RADIUS_CLEAR": str(details[0]["clear_reasons"]["guaranteed_enclosing_circle"]),
                    "Q4_RADIUS_CLEAR": str(details[1]["clear_reasons"]["guaranteed_enclosing_circle"])}
    for key, value in replacements.items():
        text = text.replace("{{"+key+"}}", value)
    (ROOT/"REPORT.md").write_text(text, encoding="utf-8")
    (ROOT/"report_data.json").write_text(json.dumps({"runs": runs, "extra_diagnostics": details}, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def main():
    (ROOT/"figures").mkdir(exist_ok=True)
    all_runs = [read_json(p) for p in sorted((ROOT/"runs").glob("*/metrics.json"))]
    runs = [r for r in all_runs if r["metadata"]["run_id"] in ("official_q3_001", "official_q4_001")]
    if len(runs) != 2 or any(not r["validation"]["all_cleared_verified"] for r in runs):
        raise ValueError("This initial report requires the two verified official practice runs")
    draw_metrics(runs)
    draw_breakdown(runs)
    draw_timeline(runs)
    draw_routes(runs)
    build_report(runs, all_runs)
    print("Generated REPORT.md, report_data.json and four PNG/SVG figure pairs.")


if __name__ == "__main__":
    main()
