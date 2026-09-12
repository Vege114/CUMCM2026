"""Freeze and verify source experiment files without changing old artifacts."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE_COMMIT = "382197175a19a044c871f27d73999715fe81f1be"
MANIFEST = Path(__file__).with_name("baseline_manifest.json")
FROZEN_PREFIXES = (
    "C题/", "data/raw/", "experiments/common/neural_v1/",
    "experiments/common/neural_v2/", "experiments/problem1/",
    "data/results/exp001/", "data/results/exp002/",
    "reports/experiments/exp001/", "reports/experiments/exp002/",
    "reports/registry/exp001.json", "reports/registry/exp002.json",
    "tests/test_neural_v1.py", "tests/test_neural_v2.py",
    "tests/test_neural_v2_cache.py", "paper/", "Xelatex/",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze():
    if MANIFEST.exists():
        return verify()
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if head != BASE_COMMIT:
        raise RuntimeError("Freeze must occur at the original v2 source commit")
    tracked = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", "-z", BASE_COMMIT], cwd=ROOT
    ).decode().split("\0")
    files = {
        name: sha256(ROOT / name)
        for name in tracked if name and name.startswith(FROZEN_PREFIXES)
    }
    if not files:
        raise RuntimeError("No protected source files found")
    MANIFEST.write_text(json.dumps({
        "base_commit": BASE_COMMIT,
        "branch": "codex/q2-optimization",
        "scope": "Existing problem statement, raw inputs, v1/v2 code/results/reports and Q1 artifacts",
        "files": files,
    }, ensure_ascii=False, indent=2) + "\n")
    return verify()


def verify():
    manifest = json.loads(MANIFEST.read_text())
    changed = [
        name for name, digest in manifest["files"].items()
        if not (ROOT / name).is_file() or sha256(ROOT / name) != digest
    ]
    if changed:
        raise RuntimeError("Frozen source files changed: " + ", ".join(changed))
    return {"base_commit": manifest["base_commit"],
            "frozen_files": len(manifest["files"]), "unchanged": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "verify"))
    args = parser.parse_args()
    print(json.dumps(freeze() if args.action == "freeze" else verify(), ensure_ascii=False))
