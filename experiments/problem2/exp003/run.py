"""Run the Q2-only experiment without invoking Q1, Q3 or Q4 stages."""

import argparse
import json
import subprocess
import sys
import time

from .data import ROOT

STAGES = ("train", "predict", "evaluate", "verify", "export", "report")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=(*STAGES, "all"))
    parser.add_argument("--run-id", default="exp003")
    parser.add_argument("--node", help="Absolute path to the bundled Codex Node executable")
    args = parser.parse_args()
    timings = ROOT / "data/results" / args.run_id / "stage_timings.json"
    timings.parent.mkdir(parents=True, exist_ok=True)
    history = json.loads(timings.read_text()) if timings.exists() else []
    for stage in STAGES if args.stage == "all" else (args.stage,):
        if stage in ("export", "report") and not args.node:
            parser.error("--node is required for workbook/report authoring")
        if stage == "report":
            subprocess.run([sys.executable, str(ROOT / "reports/q2_comparison.py")],
                           cwd=ROOT, check=True)
            command = [sys.executable, str(ROOT / "reports/build_report_q2.py"),
                       "--experiment", args.run_id, "--node", args.node]
        else:
            command = [sys.executable, "-m", f"experiments.problem2.exp003.{stage}"]
            if stage == "export":
                command += ["all", "--node", args.node]
            command += ["--run-id", args.run_id]
        began = time.monotonic()
        result = subprocess.run(command, cwd=ROOT, check=False)
        history.append({"stage": stage, "seconds": time.monotonic() - began,
                        "exit_code": result.returncode,
                        "scope": "This stage invocation; may include verified cache reuse"})
        timings.write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n")
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
