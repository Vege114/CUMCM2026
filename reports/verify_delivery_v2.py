"""Cross-check all report representations against the same measured exp002 data."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def verify(experiment="exp002"):
    report = Path(__file__).resolve().parent / "experiments" / experiment
    snapshot = json.loads((report / "app/src/data.json").read_text())
    assert snapshot["buildStatus"] == "complete"
    assert len(snapshot["reportContent"]) == 8
    assert json.loads((report / "reviewed.json").read_text()) == snapshot
    body = (report / "report.md").read_text()
    assert body.count("\n## ") == 8
    assert "仍在运行" not in body and "尚无全年汇总值" not in body
    for section in snapshot["reportContent"]:
        for block in section["blocks"]:
            if block["type"] == "prose":
                assert not any(line.startswith("|") for line in block["markdown"].splitlines())
            if block.get("queryId"):
                assert block["queryId"] in snapshot["queries"]
            if block["type"] == "table" and "rows" in block:
                assert block["rows"] == snapshot["queries"][block["queryId"]]["rows"]
                for row in block["rows"]:
                    assert "| " + " | ".join(row.values()) + " |" in body
    costs = pd.read_csv(report / "dispatch_metrics.csv", dtype={"scenario": str})
    daily = pd.read_csv(report / "daily_metrics.csv", dtype={"scenario": str})
    app_costs = pd.DataFrame(snapshot["queries"]["cost_annual"]["rows"])
    app_daily = pd.DataFrame(snapshot["queries"]["cost_daily"]["rows"])
    for source, target, keys in ((costs, app_costs, ["scenario", "name", "seed"]),
                                 (daily, app_daily, ["scenario", "name", "seed", "date"])):
        a, b = (frame.sort_values(keys).reset_index(drop=True) for frame in (source, target))
        assert a[keys].equals(b[keys])
        # CSV and JSON parsers can differ by a few ULPs near zero.
        np.testing.assert_allclose(a[["total_cost", "emergency_kwh"]], b[["total_cost", "emergency_kwh"]], rtol=1e-11, atol=1e-9)
    for row in costs.itertuples():
        source = daily[(daily.name == row.name) & (daily.scenario == row.scenario) & (daily.seed == row.seed)]
        assert len(source) == 334 and source.date.nunique() == 334
        assert source.date.min() == "2025-02-01" and source.date.max() == "2025-12-31"
        for metric in ("planned_cost", "up_cost", "down_cost", "emergency_cost", "total_cost", "emergency_kwh"):
            np.testing.assert_allclose(source[metric].sum(), getattr(row, metric), rtol=1e-11, atol=1e-7)
        if row.name == "primary" and row.seed == 42:
            assert f"{row.total_cost:,.4f}" in body
    forecast = pd.read_csv(report / "forecast_metrics.csv", dtype={"seed": str})
    january = pd.read_csv(report / "january_baselines.csv")
    assert len(january) == 9 and (january.origins == 93).all() and (january.n == 93*144).all()
    annual = pd.read_csv(report / "annual_forecast_metrics.csv", dtype={"seed": str})
    seed_results = pd.read_csv(report / "seed_forecast_results.csv", dtype={"seed": str})
    pd.testing.assert_frame_equal(seed_results, annual[annual.variant == "mlp"].reset_index(drop=True))
    seed_statistics = pd.read_csv(report / "seed_forecast_statistics.csv")
    assert len(seed_results) == 18 and len(seed_statistics) == 6
    for row in seed_statistics.itertuples():
        values = seed_results[(seed_results.target == row.target) & (seed_results.population == row.population)]
        assert set(values.seed) == {"42", "2026", "3407"} and row.seeds == 3
        for metric, prefix in (("mae", "mae"), ("rmse", "rmse"), ("wape_pct", "wape")):
            np.testing.assert_allclose([getattr(row, prefix+"_mean"), getattr(row, prefix+"_std")],
                                       [np.mean(values[metric]), np.std(values[metric], ddof=1)], rtol=1e-11)
    for row in annual.itertuples():
        source = forecast[(forecast.variant == row.variant) & (forecast.seed == row.seed) &
                          (forecast.target == row.target) & (forecast.population == row.population) & (forecast.lead == "all")]
        assert set(source.month) == set(range(2, 13))
        assert source.n.sum() == row.n
        if row.population == "all":
            assert row.n == 192168
        np.testing.assert_allclose(row.mae, source.absolute_error_sum.sum()/row.n, rtol=1e-11)
        np.testing.assert_allclose(row.rmse, np.sqrt(source.squared_error_sum.sum()/row.n), rtol=1e-11)
        np.testing.assert_allclose(row.wape_pct, 100*source.absolute_error_sum.sum()/source.actual_abs_sum.sum(), rtol=1e-11)
    specified = pd.read_csv(report / "specified_dates.csv", dtype={"scenario": str})
    assert len(specified) == 16
    assert set(specified.date) == {"2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"}
    assert len(snapshot["queries"]["specified_intervals"]["rows"]) == 16*6
    np.testing.assert_allclose(specified.total_cost, [r["total_cost"] for r in snapshot["queries"]["specified_dates"]["rows"]], rtol=1e-11)
    storage = pd.read_csv(report / "failure_storage.csv", dtype={"scenario": str})
    assert len(storage) == 4*145
    for scenario, part in storage.groupby("scenario"):
        np.testing.assert_allclose(part.hour, np.arange(145)/6)
        source = daily[(daily.scenario == scenario) & (daily.name == "primary") &
                       (daily.seed == 42) & (daily.date == part.date.iloc[0])].iloc[0]
        np.testing.assert_allclose([part.soc.iloc[0], part.soc.iloc[-1]], [source.initial_soc, source.final_soc])
    workbook = json.loads((report / "verification.json").read_text())
    for row in costs[(costs.name == "primary") & (costs.seed == 42)].itertuples():
        check = workbook[f"result{row.scenario}.xlsx"]
        np.testing.assert_allclose(check["total_cost"], row.total_cost, rtol=1e-11)
        assert check["saved_workbook_readback"] == "passed" and check["violations"] == 0
        assert hashlib.sha256((report / f"result{row.scenario}.xlsx").read_bytes()).hexdigest() == check["sha256"]
    audit = json.loads((report / "full_year_audit.json").read_text())
    assert audit["status"] == "passed" and audit["named_comparisons"] == len(costs)
    figures = ["monthly-forecast-error", "annual-three-error-metrics", "baseline-correction-decomposition",
               "training-count-and-time", "fixed-network-architecture", "scenario-information-tree",
               "three-layer-cost-comparison", "cost-components", "failure-case-and-storage",
               "cost-versus-tail-risk", "solver-gap-and-fallback", "lead-and-generating-error",
               "daily-emergency-and-monthly-cost", "worked-example-training-loss",
               "history-cost-comparison", "history-forecast-relative-change"]
    assert all((report / "figures" / f"{name}.{ext}").stat().st_size > 1000 for name in figures for ext in ("png", "svg"))
    chapter = snapshot["reportContent"][6]["blocks"]
    assert {"history-cost-chart", "history-forecast-chart", "history-training-charts"}.issubset({b["type"] for b in chapter})
    section = (report / "section-7.md").read_text()
    for name in ("history-cost-comparison", "history-forecast-relative-change", "training-count-and-time"):
        assert f"(figures/{name}.png)" in section
    bridge = pd.read_csv(report / "official_forecast_comparison.csv")
    np.testing.assert_allclose(bridge.relative_change_pct,100*(bridge.current-bridge.previous)/bridge.previous.abs(),rtol=1e-11,atol=1e-9)
    model = json.loads((report / "model_checks.json").read_text())
    assert model["formal_training_groups"] == 33 and model["gpu_output_all"]
    recovery = json.loads((report / "prediction_recovery_check.json").read_text())
    assert recovery["exact_match"] and recovery["additional_training_calls"] == 0
    archive_report = json.loads((report / "report_archive_independence_check.json").read_text())
    assert archive_report["status"] == "passed" and not archive_report["private_training_cache_required"]
    assert archive_report["full_year_results"] and archive_report["private_cache_reads_blocked_during_check"]
    causality = json.loads((report / "calibration_causality_check.json").read_text())
    assert causality["status"] == "passed" and causality["calibration_origins_across_four_questions"] == 70
    contributions = pd.read_csv(report / "core_contributions.csv")
    np.testing.assert_allclose(contributions.total_cost,
        contributions[["planned_cost", "up_cost", "down_cost", "emergency_cost"]].sum(axis=1), atol=1e-6)
    browser = json.loads((report / "browser_checks.json").read_text())
    assert browser["status"] == "passed" and browser["runtime_errors"] == []
    assert browser["history_comparison_revision"]["status"] == "passed"
    assert (report / "report.html").stat().st_size > 100000
    assert (report / "methods.md").read_text() == (report.parent.parent / "templates/methods-neural-v2.md").read_text()
    status = {"status": "passed", "sections": 8, "strategies": len(costs), "primary_scenarios": 4,
              "days_per_scenario": 334, "intervals_per_day": 144, "formal_training_groups": 33,
              "specified_dates": 16, "figures_png_and_svg": len(figures),
              "cross_checks": ["CSV ↔ reviewed snapshot", "daily ↔ annual fees", "forecast sums ↔ annual errors",
                               "Markdown ↔ formatted tables", "workbook readback ↔ annual fees", "full physical audit"],
              "browser_checks": "passed; actual interactions and offline rendering recorded in browser_checks.json"}
    (report / "consistency_checks.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="exp002")
    verify(parser.parse_args().experiment)
