"""Paired v1/v2 stratified offline suite. Failures remain in the comparison."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))

from experiments.baseline_v1.baseline.config import Config as V1Config
from experiments.baseline_v2.solver.artifacts import safe_name, source_manifest, write_json
from experiments.baseline_v2.solver.config import load_config
from experiments.baseline_v2.solver.evaluation import evaluate_run
from experiments.baseline_v2.solver.runner import execute_run
from experiments.baseline_v2.solver.scenarios import Scenario, ScenarioTransport, create_sources, scenario_hash


def run_one(folder, spec, algorithm, cfg, resume=False):
    if folder.exists():
        if not resume:
            raise FileExistsError(f"Evidence exists: {folder}; use a new prefix or explicit --resume")
        metadata = json.loads((folder/"metadata.json").read_text(encoding="utf-8"))
        expected = {"source_sha256": source_manifest()["sha256"], "config": asdict(cfg),
                    "scenario": asdict(spec), "algorithm": f"baseline_{algorithm}",
                    "source": "local_mock", "mode": "offline"}
        if any(metadata.get(k) != v for k, v in expected.items()) or metadata.get("status") == "running":
            raise ValueError(f"Cannot resume changed/incomplete evidence: {folder}")
        return metadata, evaluate_run(folder, metadata)
    sources = create_sources(spec)
    return execute_run(folder, problem=spec.problem, cfg=cfg, transport=ScenarioTransport(sources, spec),
                       algorithm=algorithm, scenario=spec, sources=sources, case_code=spec.name, quiet=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prefix", default="paired")
    p.add_argument("--manifest", type=Path, default=ROOT/"configs/scenarios.json")
    p.add_argument("--config", help="v2 settings, including ablations; v1 stays unchanged")
    p.add_argument("--scenario", action="append", help="Repeat to select cases; default all")
    p.add_argument("--resume", action="store_true", help="Reuse only matching finished evidence; never silently skip")
    args = p.parse_args(argv)
    prefix = safe_name(args.prefix)
    specs = [Scenario(**s).validate() for s in json.loads(args.manifest.read_text(encoding="utf-8-sig"))["scenarios"]]
    if len({s.name for s in specs}) != len(specs):
        p.error("Duplicate scenario names")
    if args.scenario:
        if set(args.scenario)-{s.name for s in specs}:
            p.error("Unknown selected scenario")
        specs = [s for s in specs if s.name in args.scenario]
    if not specs:
        p.error("Empty suite")
    configs = {"v1": V1Config().validate(), "v2": load_config(args.config)}
    out = ROOT/"runs"/prefix
    if out.exists() and not args.resume:
        p.error("Suite directory exists; choose another --prefix or --resume")
    out.mkdir(parents=True, exist_ok=True)
    identity = {"source_sha256": source_manifest()["sha256"], "planned_scenarios": [asdict(s) for s in specs],
                "configs": {key: asdict(cfg) for key, cfg in configs.items()}}
    if (out/"suite.json").exists():
        saved = json.loads((out/"suite.json").read_text(encoding="utf-8"))
        if any(saved.get(k) != v for k, v in identity.items()):
            p.error("Resume requires exactly the original source/configuration/scenario selection")
    suite = {"schema_version": 1, "source": "local_mock", **identity,
             "status": "running", "planned_pairs": len(specs), "rows": [],
             "note": "Self-built offline distribution; no official simulator runs or formal test opportunities."}
    passed = True
    for spec in specs:
        safe_name(spec.name)
        for algorithm in ("v1", "v2"):
            folder = out/f"{spec.name}__{algorithm}"
            metadata, metrics = run_one(folder, spec, algorithm, configs[algorithm], args.resume)
            if metadata["source_sha256"] != suite["source_sha256"]:
                raise ValueError("Source changed during the suite; use a new prefix after editing")
            valid = metrics["validation"]["experiment_passed"]
            passed &= valid
            row = {"scenario": spec.name, "problem": spec.problem, "scenario_sha256": scenario_hash(spec),
                   "source_sha256": metadata["source_sha256"],
                   "algorithm": algorithm, "run": folder.relative_to(out).as_posix(), "status": metadata["status"],
                   "passed": valid, "official_metrics": metrics["official_metrics"],
                   "diagnostics": metrics["diagnostics"], "v2_diagnostics": metrics["v2_diagnostics"]}
            suite["rows"].append(row)
            write_json(out/"suite.json", suite)
            print(f"{spec.name} {algorithm}: {metadata['status']}, "
                  f"{metadata['cleared_count']}/{metadata['total_jammers']}, "
                  f"T={metadata['virtual_total_time_s']:.2f}s", flush=True)
            if metadata.get("error", "").startswith("KeyboardInterrupt"):
                suite["status"] = "interrupted"
                write_json(out/"suite.json", suite)
                return 1
    suite["status"] = "passed" if passed else "failed"
    write_json(out/"suite.json", suite)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
