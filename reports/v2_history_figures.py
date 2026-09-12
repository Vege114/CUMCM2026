"""Paper figures for chapter 7, using the same frozen rows as the report app."""

import pandas as pd


def history_figures(out, save):
    import matplotlib.pyplot as plt

    costs = pd.read_csv(out / "dispatch_metrics.csv", dtype={"scenario": str})
    names = ["legacy_rebased", "new_deterministic", "primary", "periodic"]
    labels = ["exp001 重算", "exp002 基础", "exp002 风险", "周期基线"]
    selected = costs[(costs.seed == 42) & costs.name.isin(names)]
    limit = selected.total_cost.max() / 10000 * 1.18
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for ax, scenario in zip(axes.ravel(), ("2", "3", "4-2", "4-3")):
        rows = selected[selected.scenario == scenario].set_index("name").loc[names]
        ax.barh(labels, rows.total_cost / 10000, color=["#a4aeb6", "#5d9aae", "#247e77", "#967349"])
        for i, value in enumerate(rows.total_cost / 10000):
            ax.text(value + limit * .012, i, f"{value:,.2f}", va="center", fontsize=9)
        ax.invert_yaxis()
        ax.set(title=f"问题 {scenario}", xlabel="全年费用（万元）", xlim=(0, limit))
        ax.grid(axis="x", alpha=.15)
    fig.suptitle("exp001 重算与 exp002：相同物理及结算口径，2025 年 2—12 月")
    save(fig, "history-cost-comparison")

    bridge = pd.read_csv(out / "official_forecast_comparison.csv", dtype={"scenario": str})
    targets = {"load": "负载", "pv": "历史光伏", "pv_corrected": "预报光伏*", "price": "电价"}
    selected = bridge[(bridge.population == "all") & bridge.metric.isin(["wape_pct", "rmse"])]
    lo, hi = min(0, selected.relative_change_pct.min()), max(0, selected.relative_change_pct.max())
    span = hi - lo
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), layout="constrained", sharex=True)
    for ax, metric, title in zip(axes, ("wape_pct", "rmse"), ("WAPE", "RMSE")):
        rows = selected[selected.metric == metric]
        labels = [f"{r.scenario} · {targets[r.target]}" for r in rows.itertuples()]
        ax.barh(labels, rows.relative_change_pct, color="#2b788b" if metric == "wape_pct" else "#8666a3")
        for i, value in enumerate(rows.relative_change_pct):
            ax.text(value + (-.15 if value < 0 else .15), i, f"{value:+.2f}%",
                    ha="right" if value < 0 else "left", va="center", fontsize=9)
        ax.axvline(0, color="#777777", linewidth=.8)
        ax.invert_yaxis()
        ax.set(title=title, xlabel="相对变化（%，负值下降）", xlim=(lo - span*.2, hi + span*.2))
        ax.grid(axis="x", alpha=.15)
    fig.suptitle("exp001 → exp002：全时段预测误差；*旧发布光伏按新积分口径重算")
    save(fig, "history-forecast-relative-change")
