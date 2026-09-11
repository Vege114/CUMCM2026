"""Recompute evidence metrics; prints failures and exits nonzero for failed runs."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from experiments.baseline_v2.solver.evaluation import evaluate_run


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", nargs="?", type=Path, default=ROOT/"runs")
    args = p.parse_args(argv)
    files = sorted(args.path.rglob("metadata.json"))
    if not files:
        p.error("No run metadata found")
    passed = True
    for file in files:
        metadata = json.loads(file.read_text(encoding="utf-8-sig"))
        result = evaluate_run(file.parent, metadata)
        valid = result["validation"]["experiment_passed"]
        passed &= valid
        print(json.dumps({"run": str(file.parent), "passed": valid,
                          "errors": result["validation"]["errors"]}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
