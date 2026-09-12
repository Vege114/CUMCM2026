"""Register Q2 prediction-only evidence without rewriting historical experiments."""

import argparse
import json
import re
import subprocess

import pandas as pd

from experiments.problem2.exp004.data import HERE, OUT, ROOT, protocol, sha256, write_json
from reports.exp004_evidence import REPORT, read


def register(commit):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Pass the full commit containing the executed forecasting code")
    metadata = read(OUT / "training_metadata.json")
    for name, expected in metadata[0]["code"].items():
        content = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)
        import hashlib
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Commit does not contain measured source {name}")
    cfg = protocol()
    metrics = pd.read_csv(OUT / "dispatch_metrics.csv")
    metrics["scenario"] = "2"
    metrics["role"] = metrics.apply(
        lambda row: "future_informed_exploration" if row["exploratory"] else
        "primary" if row["name"] == cfg["primary_variant"] and row["seed"] == 42 else "ablation_or_seed", axis=1)
    record = {
        "experiment_id": "exp004", "title": "第二问预测组合与季节预判消融", "scope": ["2"],
        "protocol": cfg, "code_commit": commit, "data_hashes": metadata[0]["data"],
        "environment": metadata[0]["environment"], "seeds": cfg["seed_list"],
        "models": [cfg["architecture"]["name"]], "primary_seed": 42,
        "primary_variant": cfg["primary_variant"],
        "model_configuration": {"architecture": cfg["architecture"], "training": cfg["training"],
            "seasonality": cfg["seasonality"], "training_groups": 99, "parameters": 4930},
        "metric_definitions": {
            "mae": "sum(abs(predicted-actual))/n; kW",
            "rmse": "sqrt(sum((predicted-actual)^2)/n); kW",
            "wape_pct": "100*sum(abs(predicted-actual))/sum(abs(actual)); percent",
            "bias": "mean(predicted-actual); kW",
            "total_cost": cfg["q2_billing"] + "; yuan",
            "daily_cvar90": "weighted mean of highest 10 percent daily cost probability mass",
            "forecast_sample": "midnight 144-slot forecasts; 48096 all / 26880 PV-positive intervals",
            "comparison": "334 days; unchanged exp003 decision/execution; oracle excluded from causal ranking"},
        "forecast_metrics": json.loads(pd.read_csv(OUT / "forecast_annual.csv").to_json(orient="records", double_precision=15)),
        "metrics": json.loads(metrics.to_json(orient="records", double_precision=15)),
        "technical_path": ["Q2 input isolation", "168-hour two-branch CNN", "90-day half-life weighting",
            "tariff-weighted 5:1 Huber", "no / causal / future-informed seasonal fit", "unchanged exp003 replay"],
        "artifacts": {"report": "experiments/exp004/report.md", "html": "experiments/exp004/report.html",
            "results": "experiments/exp004/evidence", "workbooks": [f"experiments/exp004/{v}/result2.xlsx" for v in cfg["variants"]]},
        "measured_prediction_signature": read(OUT / "prediction_manifest.json")["signature"],
        "frozen_prior_manifest_sha256": sha256(HERE / "frozen_prior_manifest.json"),
        "historical_boundary_note": "Original v1 physics not directly ranked; v1/v2 four-target early stopping retained. Oracle seasonal fit includes future observations.",
    }
    schema = read(ROOT / "reports/templates/experiment.schema.json")
    for key in schema["required"]:
        assert record[key] not in (None, {}, [], ""), key
    for key in schema["properties"]["protocol"]["required"]:
        assert key in cfg, key
    destination = ROOT / "reports/registry/exp004.json"
    if destination.exists() and read(destination) != record:
        raise RuntimeError("Refusing to replace an existing different experiment record")
    write_json(destination, record)
    write_json(REPORT / "record.json", record)
    print("Registered exp004; historical records unchanged")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-commit", required=True)
    register(parser.parse_args().code_commit)
