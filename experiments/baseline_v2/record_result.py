"""Attach transcribed simulator UI results after an official run and re-evaluate."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from experiments.baseline_v2.solver.artifacts import safe_name, write_json
from experiments.baseline_v2.solver.evaluation import evaluate_run


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True, help="An existing single run name under v2/runs")
    p.add_argument("--case-code", required=True, help="Must match recorded case code")
    p.add_argument("--total", type=int, help="Practice-only source total displayed AFTER the run")
    p.add_argument("--cleared", type=int, required=True, help="Cleared count displayed by UI")
    p.add_argument("--runtime", type=float, help="Exact UI program runtime, if provided; never infer from rounded remaining time")
    p.add_argument("--evidence", type=Path, required=True, help="Local UI screenshot or redacted transcript; only name and SHA-256 are recorded")
    args = p.parse_args(argv)
    folder = ROOT/"runs"/safe_name(args.name)
    metadata = json.loads((folder/"metadata.json").read_text(encoding="utf-8-sig"))
    if metadata.get("source") != "official_simulator" or metadata.get("status") == "running":
        p.error("Only a finished official-backend run can receive UI results")
    if metadata.get("case_code") != args.case_code:
        p.error("Case code does not match this run")
    if not 0 <= args.cleared <= 16:
        p.error("Cleared count must be 0..16")
    if args.cleared != metadata.get("cleared_count"):
        p.error("UI count differs from the public log; retain evidence and investigate before changing metadata")
    if args.total is not None and (metadata.get("mode") != "practice" or not 10 <= args.total <= 16 or args.total < args.cleared):
        p.error("Total is only available for practice and must be 10..16, no smaller than cleared")
    if args.runtime is not None and not (math.isfinite(args.runtime) and 0 <= args.runtime <= 1200):
        p.error("Invalid program runtime")
    evidence = {"file_name": args.evidence.name, "sha256": hashlib.sha256(args.evidence.read_bytes()).hexdigest(),
                "provenance": "user_transcribed_simulator_ui"}
    metadata.setdefault("ui_result_history", []).append({"evidence": evidence, "cleared": args.cleared,
                                                          "total": args.total, "runtime": args.runtime})
    metadata["official_cleared_count"] = args.cleared
    if args.total is not None:
        metadata.update(total_jammers=args.total, total_source="simulator_ui_post_run")
    if args.runtime is not None:
        metadata.update(program_runtime_s=args.runtime, program_runtime_source="simulator_ui")
    write_json(folder/"metadata.json", metadata)
    result = evaluate_run(folder, metadata)
    print(json.dumps({"run": str(folder), **result["official_metrics"]}, ensure_ascii=False))
    return 0 if result["validation"]["experiment_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
