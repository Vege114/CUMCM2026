"""Register verified Q2 planning results without rewriting historical records."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/experiments/exp006"


def read(path):
    return json.loads(path.read_text())


def register(commit):
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Use the full commit containing the executed source")
    manifest = read(ROOT / "data/results/exp006/run_manifest.json")
    if not manifest["complete"] or len(manifest["completed_runs"]) != 15:
        raise ValueError("All approved runs must be complete before registration")
    for name, expected in manifest["evidence"]["source_sha256"].items():
        content = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Commit does not contain the executed source: {name}")
    record = read(REPORT / "record.draft.json")
    record["code_commit"] = commit
    record["code_commit_status"] = "verified against all executed source hashes"
    record["formal_run_executed"] = True
    record["protocol_snapshot_note"] = (
        "protocol.formal_run_executed is the immutable approval-time snapshot; "
        "the completed run_manifest and this record describe final execution."
    )
    schema = read(ROOT / "reports/templates/experiment.schema.json")
    for key in schema["required"]:
        if record.get(key) in (None, {}, [], ""):
            raise ValueError(f"Missing record field: {key}")
    for key in schema["properties"]["protocol"]["required"]:
        if key not in record["protocol"]:
            raise ValueError(f"Missing comparison protocol field: {key}")
    for artifact in ("report.md", "report.html", "result2.xlsx",
                     "greedy_execution/result2.xlsx"):
        if not (REPORT / artifact).is_file():
            raise ValueError(f"Missing delivery artifact: {artifact}")
    destination = ROOT / "reports/registry/exp006.json"
    if destination.exists() and read(destination) != record:
        raise ValueError("Refusing to overwrite a different registered experiment")
    encoded = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    destination.write_text(encoded)
    (REPORT / "record.json").write_text(encoded)
    print(f"Registered exp006 with source commit {commit}; prior records unchanged")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-commit", required=True)
    register(parser.parse_args().code_commit)
