"""Seven independently resumable stages for the fixed-network experiment."""

import argparse
import json
import subprocess
import sys
import time

from .data import HERE, ROOT, Data, protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "train", "predict", "calibrate",
                                          "replay", "export", "report", "all"))
    parser.add_argument("--run-id", default="exp002")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--node", help="Bundled Codex Node executable for workbook export")
    args = parser.parse_args()
    directory = HERE / "runs" / args.run_id
    directory.mkdir(parents=True, exist_ok=True)
    stages = ("prepare", "train", "predict", "calibrate", "replay", "export", "report")
    timings_path = directory / "stage_timings.json"
    timings = json.loads(timings_path.read_text()) if timings_path.exists() else []
    for stage in stages if args.stage == "all" else (args.stage,):
        start = time.monotonic()
        if stage in ("predict", "calibrate", "replay", "export", "report"):
            from .manifest import validate_upstream
            validate_upstream(args.run_id, replay=stage in ("export", "report"))
        if stage == "prepare":
            data = Data()
            assert data.actual.shape == (52560, 3)
            output = ROOT / "data/results" / args.run_id
            output.mkdir(parents=True, exist_ok=True)
            (output / "data_hashes.json").write_text(json.dumps(data.hashes, indent=2))
            (output / "protocol.json").write_text(json.dumps(protocol(), indent=2))
        elif stage == "train":
            from .train import run
            run(args.run_id)
        elif stage == "predict":
            from .predict import run
            run(args.run_id)
            from .manifest import run as manifest
            manifest(args.run_id)
        elif stage == "calibrate":
            from .evaluate import calibrate
            calibrate(args.run_id, args.workers)
        elif stage == "replay":
            from .evaluate import run
            run(args.run_id, args.workers)
        elif stage == "export":
            from .export import run
            run(args.run_id, args.node)
        elif stage == "report":
            command = [sys.executable, str(ROOT / "reports/build_report_v2.py"), "--experiment", args.run_id]
            if args.node:
                command.extend(["--node", args.node])
            subprocess.run(command, check=True, cwd=ROOT)
        timings.append({"stage": stage, "seconds": time.monotonic() - start})
        timings_path.write_text(json.dumps(timings, indent=2))


if __name__ == "__main__":
    main()
