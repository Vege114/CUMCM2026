"""Lifecycle, robust cleanup and immediate independent log validation."""
from contextlib import redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
import io
import math
import sys
import time
import traceback

from experiments.baseline_v1.baseline.strategy import Strategy as V1Strategy, BudgetReached
from .artifacts import git_commit, source_manifest, write_json
from .evaluation import evaluate_run
from .protocol import CheckedTransport, Client
from .strategy import Strategy


def execute_run(folder, *, problem, cfg, transport, robot_id="OFFLINE", algorithm="v2",
                mode="offline", case_code=None, scenario=None, sources=None, quiet=False):
    if algorithm not in {"v1", "v2"}:
        raise ValueError("Unknown algorithm")
    if (mode == "offline") != (sources is not None):
        raise ValueError("Truth fixtures are only permitted for offline runs")
    folder.mkdir(parents=True, exist_ok=False)
    manifest = source_manifest()
    metadata = {"run_id": folder.name, "problem": problem, "mode": mode,
                "source": "local_mock" if mode == "offline" else "official_simulator",
                "algorithm": f"baseline_{algorithm}", "case_code": case_code,
                "config": asdict(cfg), "git_commit": git_commit(), "source_sha256": manifest["sha256"],
                "source_manifest": manifest["files"], "created_at": datetime.now(timezone.utc).isoformat(),
                "python_version": sys.version, "platform": sys.platform, "status": "running",
                "total_jammers": None, "total_source": None,
                "scenario": asdict(scenario) if scenario else None,
                "seed": scenario.seed if scenario else None}
    write_json(folder/"metadata.json", metadata)
    client = Client(CheckedTransport(transport), robot_id, folder/"events.jsonl", cfg.http_retries)
    strategy = (Strategy if algorithm == "v2" else V1Strategy)(client, cfg, problem, folder/"decisions.jsonl")
    started = time.monotonic()
    def cleanup():
        if client.active and not client.uncertain:
            try:
                client.call("/exit")
            except Exception as error:
                metadata["exit_failed"] = type(error).__name__
    try:
        before_enter = time.monotonic()
        entered = client.call("/enter")
        # Retries may replay an older enter response. Subtract all elapsed time,
        # rather than granting a new full budget upon recovery of its response.
        remaining = max(0.0, float(entered["remaining_real_duration_s"])-(time.monotonic()-before_enter))
        if not math.isfinite(remaining):
            raise ValueError("Non-finite remaining duration")
        with redirect_stdout(io.StringIO()) if quiet else redirect_stdout(sys.stdout):
            reason = strategy.run(remaining)
        client.call("/exit")
        metadata.update(status="completed", stop_reason=reason)
    except BudgetReached as error:
        metadata.update(status="incomplete", stop_reason="budget", error=str(error))
        cleanup()
    except (Exception, KeyboardInterrupt) as error:
        if isinstance(error, KeyboardInterrupt):
            # Interruption may arrive while a request is executing. Do not send a
            # new action before its outcome is known; the UI can end the session.
            client.uncertain = True
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}".replace(robot_id, "REDACTED"))
        (folder/"error.txt").write_text(traceback.format_exc().replace(robot_id, "REDACTED"), encoding="utf-8")
        cleanup()
    finally:
        metadata.update(client_runtime_s=time.monotonic()-started, cleared_count=strategy.cleared_count,
                        virtual_total_time_s=client.virtual_time_s, actions=strategy.actions,
                        survey_points_visited=len(strategy.survey_visited), outcome_uncertain=client.uncertain)
        if sources is not None:
            metadata.update(total_jammers=len(sources), total_source="local_mock_post_run_truth",
                            directional_jammers=sum(s.orientation_deg is not None for s in sources))
            write_json(folder/"offline_truth.json", [asdict(s) for s in sources])
        client.close()
        strategy.close()
        write_json(folder/"metadata.json", metadata)
    metrics = evaluate_run(folder, metadata)
    if metadata["status"] == "completed" and not metrics["validation"]["experiment_passed"]:
        metadata.update(status="failed_validation", error="Independent replay or full-clear check failed")
        write_json(folder/"metadata.json", metadata)
        metrics = evaluate_run(folder, metadata)
    return metadata, metrics
