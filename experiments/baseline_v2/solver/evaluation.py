"""Reuse independent Benchmark replay, plus v2 diagnostic summaries."""
from collections import Counter

from Benchmark.evaluate import evaluate, read_jsonl
from .artifacts import write_json


def evaluate_run(folder, metadata):
    result = evaluate(read_jsonl(folder/"events.jsonl"), metadata)
    decisions = read_jsonl(folder/"decisions.jsonl")
    counts = Counter(d.get("event") for d in decisions)
    timeline = result["clearance_timeline"]
    last_clear = timeline[-1]["virtual_time_s"] if timeline else None
    total = result["official_metrics"]["virtual_total_time_s"]
    result["v2_diagnostics"] = {
        "optical_fallback_count": counts["optical_fallback"],
        "opportunistic_survey_count": counts["opportunistic_survey"],
        "absence_certificate_count": counts["absence_certificate"],
        "post_last_clear_time_s": total-last_clear if total is not None and last_clear is not None else None,
        "stop_certificate_count": counts["stop_certificate"],
    }
    # Benchmark valid_run means a well-formed complete log, not strategy success.
    must_verify_total = (metadata.get("source") == "local_mock" or
                         metadata.get("mode") != "formal" and metadata.get("total_jammers") is not None)
    result["validation"]["experiment_passed"] = (
        metadata.get("status") == "completed" and result["validation"]["valid_run"]
        and (not must_verify_total or result["validation"]["all_cleared_verified"]))
    write_json(folder/"metrics.json", result)
    return result
