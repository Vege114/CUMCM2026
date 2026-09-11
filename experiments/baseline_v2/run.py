"""Run from any working directory; --dry-run never sends simulator requests."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import unicodedata

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))

from experiments.baseline_v1.baseline.config import Config as V1Config
from experiments.baseline_v2.solver.artifacts import safe_name, source_manifest
from experiments.baseline_v2.solver.config import load_config
from experiments.baseline_v2.solver.coverage import survey_points
from experiments.baseline_v2.solver.protocol import HttpTransport
from experiments.baseline_v2.solver.runner import execute_run
from experiments.baseline_v2.solver.scenarios import Scenario, ScenarioTransport, create_sources


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("mock", "official"), default="mock")
    p.add_argument("--problem", type=int, choices=(3, 4), default=3)
    p.add_argument("--algorithm", choices=("v1", "v2"), default="v2")
    p.add_argument("--seed", type=int, default=20260910)
    p.add_argument("--scenario", help="Name from configs/scenarios.json; determines problem and seed")
    p.add_argument("--config")
    p.add_argument("--name")
    p.add_argument("--mode", choices=("practice", "formal"), help="Required for official backend; must match simulator UI")
    p.add_argument("--case-code")
    p.add_argument("--base-url", default="http://127.0.0.1:2026")
    p.add_argument("--dry-run", action="store_true", help="Validate configuration/imports only; no HTTP, no evidence directory")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.algorithm == "v1" and args.config:
        p.error("v1 paired runs use unchanged v1 default configuration")
    cfg = load_config(args.config) if args.algorithm == "v2" else V1Config().validate()
    scenario = None
    if args.backend == "mock":
        if args.mode or args.case_code:
            p.error("--mode/--case-code are official-only fields")
        if args.scenario:
            specs = json.loads((ROOT/"configs/scenarios.json").read_text(encoding="utf-8"))["scenarios"]
            matches = [s for s in specs if s["name"] == args.scenario]
            if not matches:
                p.error("Unknown offline scenario")
            scenario = Scenario(**matches[0]).validate()
        else:
            scenario = Scenario(f"legacy_q{args.problem}_{args.seed}", args.problem, args.seed).validate()
        args.problem = scenario.problem
    else:
        if args.scenario:
            p.error("Offline scenarios cannot be passed to official backend")
        if not args.mode or not args.case_code:
            p.error("Official backend requires --mode and --case-code copied from simulator UI")
        HttpTransport(args.base_url, cfg.http_timeout_s)  # Validate without connecting.
    name = safe_name(args.name or datetime.now().strftime("%Y%m%d_%H%M%S_%f")+f"_q{args.problem}_{args.backend}")
    if args.dry_run:
        from experiments.baseline_v2.solver.config import Config
        plan_cfg = cfg if args.algorithm == "v2" else Config(triangular_coverage=False, survey_ring_m=1400)
        print(json.dumps({"ready": True, "backend": args.backend, "problem": args.problem,
                          "algorithm": args.algorithm, "http_requests_sent": 0,
                          "survey_points": len(survey_points(args.problem, plan_cfg)),
                          "source_sha256": source_manifest()["sha256"]}, ensure_ascii=False))
        return 0
    robot_id = "OFFLINE"
    if args.backend == "official":
        if sys.platform != "win32":
            p.error("Official exe backend runs on Windows; use --dry-run or --backend mock here")
        robot_id = os.environ.get("JAMMERS_ROBOT_ID", "")
        if not 1 <= len(robot_id.encode("utf-8")) <= 64 or any(unicodedata.category(c).startswith("C") for c in robot_id):
            p.error("Set JAMMERS_ROBOT_ID to the currently logged-in team ID (never stored)")
    sources = create_sources(scenario) if scenario else None
    transport = ScenarioTransport(sources, scenario) if scenario else HttpTransport(args.base_url, cfg.http_timeout_s)
    metadata, metrics = execute_run(ROOT/"runs"/name, problem=args.problem, cfg=cfg, transport=transport,
                                   robot_id=robot_id, algorithm=args.algorithm,
                                   mode=args.mode if args.backend == "official" else "offline",
                                   case_code=args.case_code if scenario is None else scenario.name,
                                   scenario=scenario, sources=sources)
    print(json.dumps({"run_dir": str(ROOT/"runs"/name), "status": metadata["status"],
                      **metrics["official_metrics"]}, ensure_ascii=False))
    return 0 if metrics["validation"]["experiment_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
