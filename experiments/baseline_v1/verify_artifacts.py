"""Check stored evidence, source fingerprints and report links without new runs."""
import hashlib
import json
from pathlib import Path
import re
from run import ROOT, source_fingerprint


def main():
    records = []
    code_hash = source_fingerprint()
    for path in sorted((ROOT/"runs").glob("*/metadata.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        metrics = json.loads((path.parent/"metrics.json").read_text(encoding="utf-8"))
        assert metrics["validation"]["valid_run"], path
        assert metrics["validation"]["all_cleared_verified"], path
        assert meta["source_sha256"] == code_hash, f"Source changed since {path.parent.name}"
        events = [json.loads(line) for line in (path.parent/"events.jsonl").read_text(encoding="utf-8").splitlines()]
        assert all(e["request"]["robot_id"] == "REDACTED" for e in events), path
        if meta["source"] == "official_simulator":
            ui = (path.parent/"ui_evidence.txt").read_text(encoding="utf-8")
            assert meta["case_code"] in ui
            assert f"共{meta['total_jammers']}个" in ui
            assert "队号 REDACTED" in ui
        records.append({"run_id": meta["run_id"], "events_sha256_lf_utf8": hashlib.sha256((path.parent/"events.jsonl").read_text(encoding="utf-8").encode("utf-8")).hexdigest(),
                        "time_balance_error_s": metrics["diagnostics"]["time_balance_error_s"], "verified": True})
    report = (ROOT/"REPORT.md").read_text(encoding="utf-8")
    assert "{{" not in report, "Unexpanded report placeholder"
    for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", report):
        if not target.startswith("http"):
            assert (ROOT/target).exists(), target
    result = {"source_sha256": code_hash, "run_count": len(records), "runs": records,
              "tests_observed": {"model_tests": 11, "benchmark_evaluator_tests": 7,
                                 "model_command": "python -m unittest discover -s experiments/baseline_v1/tests -v",
                                 "evaluator_command": "python -m unittest discover -s Benchmark -p test_*.py -v",
                                 "status": "passed during implementation; artifact audit does not re-run them"},
              "figures_visually_inspected": ["official_metrics.png", "time_breakdown.png", "clearance_timeline.png", "routes.png"]}
    (ROOT/"validation_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"verified_runs": len(records), "source_match": True, "redaction": True, "report_links": True}))


if __name__ == "__main__":
    main()
