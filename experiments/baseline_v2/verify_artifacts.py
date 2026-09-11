"""Independent post-run truth/geometry audit, never imported by the strategy."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from Benchmark.evaluate import evaluate, read_jsonl
from experiments.baseline_v2.solver.artifacts import source_manifest


def inside(point, vertices):
    # Independently computed oriented edge distances, with length-scaled tolerance.
    if len(vertices) == 1:
        return math.dist(point, vertices[0]) <= 1e-5
    if len(vertices) == 2:
        return abs(math.dist(point, vertices[0])+math.dist(point, vertices[1])-
                   math.dist(*vertices)) <= 1e-5
    return bool(vertices) and all((b[0]-a[0])*(point[1]-a[1])-(b[1]-a[1])*(point[0]-a[0])
                                 >= -1e-6*max(1, math.dist(a, b))
                                 for a, b in zip(vertices, vertices[1:]+vertices[:1]))


def audit(folder, require_current_source=False):
    metadata = json.loads((folder/"metadata.json").read_text(encoding="utf-8-sig"))
    events = read_jsonl(folder/"events.jsonl")
    result = evaluate(events, metadata)
    errors = list(result["validation"]["errors"])
    source_matches = metadata.get("source_sha256") == source_manifest()["sha256"]
    if require_current_source and not source_matches:
        errors.append("recorded source/config manifest differs from current checkout")
    if not result["validation"]["valid_run"]:
        errors.append("missing/invalid complete event log")
    if metadata.get("status") != "completed":
        errors.append("strategy did not complete")
    truth_path = folder/"offline_truth.json"
    truth = {s["channel"]: s for s in json.loads(truth_path.read_text(encoding="utf-8"))} if truth_path.exists() else {}
    if metadata["source"] == "local_mock" and (not truth or len(truth) != metadata.get("total_jammers")):
        errors.append("missing/inconsistent offline truth artifact")
    if metadata["source"] == "local_mock" and not result["validation"]["all_cleared_verified"]:
        errors.append("offline full-clear verification failed")
    if result["official_metrics"]["total_jammers"] is not None and not result["validation"]["all_cleared_verified"]:
        errors.append("known source total does not support verified full clearance")
    polygons, updates = {}, 0
    for d in read_jsonl(folder/"decisions.jsonl"):
        ch = d.get("channel")
        if d["event"] == "bearing_update":
            updates += 1
            polygons[ch] = d["vertices"]
            if ch in truth and not inside(truth[ch]["position"], d["vertices"]):
                errors.append(f"channel {ch}: truth excluded by feasible polygon")
        if d["event"] == "clear_attempt" and d.get("reason") == "guaranteed_enclosing_circle":
            if ch not in polygons or max(math.dist(d["position"], q) for q in polygons[ch]) > 20+1e-5:
                errors.append(f"channel {ch}: invalid enclosing-circle certificate")
        if d["event"] == "absence_certificate" and ch in truth:
            errors.append(f"channel {ch}: a real source was incorrectly declared absent")
    if any(e.get("request", {}).get("robot_id") != "REDACTED" for e in events):
        errors.append("unredacted robot_id")
    return {"run": folder.name, "passed": not errors, "source_matches_current": source_matches, "bearing_updates_checked": updates,
            "errors": sorted(set(errors))}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--require-current-source", action="store_true")
    args = p.parse_args(argv)
    rows = [audit(f.parent, args.require_current_source) for f in sorted(args.path.rglob("metadata.json"))]
    if not rows:
        p.error("No runs found")
    summary = {"passed": all(r["passed"] for r in rows), "runs": len(rows), "rows": rows}
    if args.output:
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"passed": summary["passed"], "runs": len(rows),
                      "failures": [r for r in rows if not r["passed"]]}, ensure_ascii=False))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
