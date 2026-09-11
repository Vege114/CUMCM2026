"""Rebuild v2 report figures from archived evidence; never modifies solver sources."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v1_report_figures", ROOT.parent / "baseline_v1/make_report.py")
plots = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plots)
# Reuse the established visual style, redirecting every output and input to v2.
plots.ROOT = ROOT


def draw_routes(runs):
    fig, axes = plots.plt.subplots(1, 2, figsize=(11.5, 5.8), layout="constrained")
    all_paths = []
    for run in runs:
        path, clears = [(0, 0)], []
        for event in plots.read_jsonl(ROOT / "runs" / run["metadata"]["run_id"] / "events.jsonl"):
            req, resp = event["request"], event["response"] or {}
            if resp.get("accepted") and "position" in req:
                p = (req["position"]["x"], req["position"]["y"])
                if p != path[-1]:
                    path.append(p)
                if resp.get("clear_result") == "success":
                    clears.append((p, req["channel"]))
        all_paths.append((path, clears))
    limit = max(1800, *(abs(v) for path, _ in all_paths for p in path for v in p)) * 1.12
    for ax, run, (path, clears), color in zip(axes, runs, all_paths, plots.COLORS):
        ax.plot(*zip(*path), color=color, linewidth=1, alpha=.8)
        ax.add_patch(plots.Circle((0, 0), 1800, fill=False, linestyle="--", color="#7B8690", linewidth=1))
        ax.scatter([0], [0], marker="s", color="#202A32", s=35, label="起点")
        ax.scatter(*zip(*(p for p, _ in clears)), marker="*", s=55, color="#BB5055", zorder=4, label="成功清除位置（标号为频道）")
        ax.set(title=f"Q{run['metadata']['problem']}：{run['diagnostics']['travel_distance_m']/1000:.2f} km", xlabel="东向坐标 x（米）", ylabel="北向坐标 y（米）", xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
        ax.grid(alpha=.15)
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.14), frameon=False, fontsize=9)
        fig.canvas.draw()
        occupied = []
        for p, channel in clears:
            label = ax.annotate(str(channel), p, xytext=(6, 6), textcoords="offset points", fontsize=9)
            for offset in [(6, 6), (6, -14), (-18, 6), (-18, -14), (6, 20), (-18, 20)]:
                label.set_position(offset)
                bbox = label.get_window_extent(fig.canvas.get_renderer()).expanded(1.2, 1.2)
                if not any(bbox.overlaps(b) for b in occupied):
                    break
            occupied.append(bbox)
    plots.save(fig, "routes")


def main():
    (ROOT / "figures").mkdir(exist_ok=True)
    runs = [plots.read_json(ROOT / "runs" / f"official_q{p}_001" / "metrics.json") for p in (3, 4)]
    for run in runs:
        if not run["validation"]["experiment_passed"] or not run["validation"]["all_cleared_verified"]:
            raise ValueError("Report requires verified complete practice runs")
    plots.draw_metrics(runs)
    plots.draw_breakdown(runs)
    plots.draw_timeline(runs)
    draw_routes(runs)
    suite = plots.read_json(ROOT / "runs/validation_20260911/suite.json")
    fig, axes = plots.plt.subplots(1, 2, figsize=(11.5, 4.8), layout="constrained")
    for ax, problem in zip(axes, (3, 4)):
        pairs = {}
        for row in suite["rows"]:
            if row["problem"] == problem:
                if not row["passed"]:
                    raise ValueError("A failed pair must not be silently omitted")
                pairs.setdefault(row["scenario"], {})[row["algorithm"]] = row["official_metrics"]["average_localization_clear_time_s"]
        for pair in pairs.values():
            ax.plot([0, 1], [pair["v1"], pair["v2"]], color=plots.COLORS[1] if pair["v2"] > pair["v1"] else plots.COLORS[0], marker="o", alpha=.65, linewidth=1.2)
        avg = [sum(p[a] for p in pairs.values())/len(pairs) for a in ("v1", "v2")]
        ax.plot([0, 1], avg, "k--", linewidth=2, label="各局均值")
        for x, y in enumerate(avg):
            ax.annotate(f"均值 {y:.2f}", (x, y), xytext=(10 if x == 0 else -10, 10), textcoords="offset points", ha="left" if x == 0 else "right")
        ax.set(title=f"Q{problem}：{len(pairs)} 对自建场景", xticks=[0, 1], xticklabels=["v1", "v2"], ylabel="整场平均定位清除时间（虚拟秒/个）", xlim=(-.15, 1.15), ylim=(0, None))
        ax.grid(axis="y", alpha=.18)
        ax.legend(frameon=False)
    fig.suptitle("离线同场景配对：每条实线是一对，橙色保留退步案例")
    plots.save(fig, "offline_pairs")
    (ROOT / "report_data.json").write_text(json.dumps({"runs": runs, "extra_diagnostics": [plots.diagnostic_details(r) for r in runs], "offline_suite": "runs/validation_20260911/suite.json"}, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print("Generated five PNG/SVG figure pairs and report_data.json")


if __name__ == "__main__":
    main()
