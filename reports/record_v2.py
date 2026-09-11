"""Create an explicit official-policy record without mutating the registry."""

import json
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def make_record(experiment="exp002", code_commit=None):
    out = ROOT / "data/results" / experiment
    registered = ROOT / "reports/registry" / f"{experiment}.json"
    if code_commit is None and registered.exists():
        code_commit = json.loads(registered.read_text())["code_commit"]
    commit = code_commit or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    protocol = json.loads((out / "protocol.json").read_text())
    training = json.loads((out / "training_metadata.json").read_text())
    annual = pd.read_csv(out / "annual_forecast_metrics.csv", dtype={"seed": str})
    forecasts = annual[annual.seed == "none"].to_dict("records")
    for scenario in ("2", "3", "4-2", "4-3"):
        targets = ["load", "pv_corrected" if scenario in ("3", "4-3") else "pv"]
        if scenario.startswith("4"):
            targets.append("price")
        for row in annual[(annual.seed == "42") & annual.target.isin(targets)].to_dict("records"):
            forecasts.append({**row, "role": "primary", "scenario": scenario})
    costs = pd.read_csv(out / "dispatch_metrics.csv", dtype={"scenario": str})
    primary = costs[(costs.name == "primary") & (costs.seed == 42)]
    record = {
        "experiment_id": experiment, "title": "固定轻量网络与多阶段风险调度",
        "protocol": protocol, "code_commit": commit,
        "data_hashes": json.loads((out / "data_hashes.json").read_text()),
        "environment": training[0]["environment"], "seeds": [42, 2026, 3407],
        "models": ["periodic_residual_mlp"], "primary_seed": 42,
        "model_configuration": {"architecture": protocol["architecture"], "training": protocol["training"],
                                 "risk": protocol["risk"], "formal_training_groups": 33},
        "metric_definitions": {
            "mae": "sum(abs(predicted-actual))/n",
            "rmse": "sqrt(sum((predicted-actual)^2)/n)",
            "wape_pct": "100*sum(abs(predicted-actual))/sum(abs(actual))",
            "forecast_sample": "all four daily issues, 144 future intervals, observed labels only",
            "total_cost": "sum(price*(original+1.5*max(final-original,0)+0.5*max(original-final,0)+5*emergency))",
            "daily_cvar90": "weighted mean of highest 10 percent daily cost probability mass, including fractional boundary day"},
        "forecast_metrics": forecasts,
        "metrics": json.loads(primary.to_json(orient="records", double_precision=12)),
        "technical_path": ["区间终点与小时点积分", "周期基础预测", "固定四分支轻量残差网络",
                           "固定正式种子与三种子稳定性评价", "联合残差与非预见性场景树",
                           "期望费用加条件尾部风险的混合整数规划", "精确因果储能执行", "最终净调整结算"],
        "historical_replay": json.loads((out / "prediction_archive.json").read_text())["historical_replay"],
        "artifacts": {"report": f"experiments/{experiment}/report.md", "html": f"experiments/{experiment}/report.html",
                      "results": f"experiments/{experiment}", "replay_arrays_on_experiment_branch": f"../data/results/{experiment}",
                      "audit": f"experiments/{experiment}/full_year_audit.json"}}
    (ROOT / "reports/experiments" / experiment / "record.draft.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False))
    return record
