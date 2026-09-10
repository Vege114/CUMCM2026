"""Run from any directory; all artifacts stay alongside this baseline."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from baseline.config import load_config
from baseline.mock import MockTransport, generate_sources
from baseline.protocol import Client, HttpTransport
from baseline.strategy import BudgetReached, Strategy

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def source_fingerprint():
    digest = hashlib.sha256()
    for path in sorted((ROOT/"baseline").glob("*.py")) + [ROOT/"run.py"]:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run(args):
    cfg = load_config(args.config)
    name = args.name or datetime.now().strftime("%Y%m%d_%H%M%S")+f"_q{args.problem}_{args.backend}"
    out = ROOT / "runs" / name
    out.mkdir(parents=True, exist_ok=False)  # Never overwrite evidence from an earlier run.
    robot_id = os.environ.get("JAMMERS_ROBOT_ID", "")
    if args.backend == "official" and not robot_id:
        raise ValueError("Set JAMMERS_ROBOT_ID to the currently logged-in team identifier")
    sources = generate_sources(args.seed, args.problem) if args.backend == "mock" else None
    transport = (MockTransport(sources, args.seed) if sources is not None
                 else HttpTransport(args.base_url, cfg.http_timeout_s))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, text=True, capture_output=True).stdout.strip()
    metadata = {"run_id": name, "problem": args.problem,
                "mode": "offline" if args.backend == "mock" else "practice",
                "source": "local_mock" if args.backend == "mock" else "official_simulator",
                "case_code": f"mock-q{args.problem}-seed{args.seed}" if sources else args.case_code,
                "total_jammers": None, "total_source": None, "algorithm": "baseline_v1",
                "config": asdict(cfg), "git_commit": commit, "source_sha256": source_fingerprint(),
                "created_at": datetime.now(timezone.utc).isoformat(), "python_version": sys.version,
                "seed": args.seed if sources else None, "status": "running"}
    write_json(out/"metadata.json", metadata)
    client = Client(transport, robot_id or "OFFLINE", out/"events.jsonl", cfg.http_retries)
    strategy = Strategy(client, cfg, args.problem, out/"decisions.jsonl")
    started = time.monotonic()
    try:
        entered = client.call("/enter")
        reason = strategy.run(float(entered["remaining_real_duration_s"]))
        client.call("/exit")
        metadata.update(status="completed", stop_reason=reason)
    except BudgetReached as error:
        metadata.update(status="incomplete", stop_reason="budget", error=str(error))
        if client.active and not client.uncertain:
            client.call("/exit")
    except Exception as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        (out/"error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        # Unknown transport outcome may mean the test ended; do not query /exit.
        if client.active and not client.uncertain:
            try:
                client.call("/exit")
            except Exception:
                metadata["exit_failed"] = True
    finally:
        metadata.update(client_runtime_s=time.monotonic()-started, cleared_count=strategy.cleared_count,
                        virtual_total_time_s=client.virtual_time_s, actions=strategy.actions,
                        survey_points_visited=len(strategy.survey_visited))
        if sources is not None:
            metadata.update(total_jammers=len(sources), total_source="local_mock_post_run_truth",
                            directional_jammers=sum(s.orientation_deg is not None for s in sources))
            write_json(out/"offline_truth.json", [asdict(s) for s in sources])
        client.close()
        strategy.close()
        write_json(out/"metadata.json", metadata)
    print(json.dumps({"run_dir": str(out), **{k: metadata[k] for k in
          ("status", "cleared_count", "virtual_total_time_s", "client_runtime_s")}}, ensure_ascii=False))
    return 0 if metadata["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mock", "official"), default="mock")
    parser.add_argument("--problem", choices=(3, 4), type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--name")
    parser.add_argument("--case-code")
    parser.add_argument("--config")
    parser.add_argument("--base-url", default="http://127.0.0.1:2026")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
