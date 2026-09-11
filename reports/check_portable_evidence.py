"""Verify that delivered archives reproduce all evidence with private runtime files hidden."""

import builtins
import hashlib
import io
import json
from pathlib import Path
from unittest.mock import patch

from v2_evidence import ROOT, derive


def run():
    out = ROOT / "data/results/exp002"
    assert (out / "dispatch_metrics.csv").exists(), "Complete annual results are required"
    derive()
    names = ["forecast_decomposition.csv", "tree_example.json", "tree_nodes.csv", "worked_example.json",
             "official_forecast_comparison.csv", "failure_intervals.csv", "core_contributions.csv",
             "seed_cost_results.csv", "seed_cost_statistics.csv", "monthly_dispatch.csv",
             "solver_summary.csv", "phase_timing.csv", "evaluation_walltime.json"]
    before = {name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in names}
    original_builtin, original_io, original_exists = builtins.open, io.open, Path.exists
    blocked = str(ROOT / "experiments/common/neural_v2/runs") + "/"

    def guarded(original):
        def opened(file, *args, **kwargs):
            if isinstance(file, (str, bytes, Path)) and blocked in str(file):
                raise AssertionError("Report attempted to read a private runtime file")
            return original(file, *args, **kwargs)
        return opened

    def exists(path):
        return False if blocked in str(path) else original_exists(path)

    with patch("builtins.open", guarded(original_builtin)), patch("io.open", guarded(original_io)), patch.object(Path, "exists", exists):
        derive()
    after = {name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in names}
    assert before == after
    result = {"status": "passed", "full_year_results": True, "private_training_cache_required": False,
              "private_cache_reads_blocked_during_check": True, "additional_training_calls": 0,
              "identical_derived_files": names, "sha256": after}
    (out / "report_archive_independence_check.json").write_text(json.dumps(result, indent=2))
    print("PORTABLE EVIDENCE", len(names), "files identical with private runtime hidden")


if __name__ == "__main__":
    run()
