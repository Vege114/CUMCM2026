"""Evidence and source provenance shared by single runs and paired suites."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def safe_name(name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}", name):
        raise ValueError("Name must use 1..101 ASCII letters/digits, underscore, dot or hyphen")
    if name.endswith(".") or name.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *[f"{p}{i}" for p in ("COM", "LPT") for i in range(1, 10)]}:
        raise ValueError("Name is reserved by Windows")
    return name


def source_manifest():
    files = list((ROOT/"solver").glob("*.py")) + list(ROOT.glob("*.py")) + list(ROOT.glob("*.ps1"))
    files += list((ROOT/"configs").glob("*.json"))
    files += list((REPO/"experiments/baseline_v1/baseline").glob("*.py"))
    files += [REPO/"Benchmark/evaluate.py"]
    # Git checks PowerShell out with CRLF on Windows. Normalize text newlines so
    # the same committed sources have the same identity on macOS and Windows.
    entries = {p.relative_to(REPO).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
               for p in sorted(files)}
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"sha256": digest, "files": entries}


def git_commit():
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, timeout=5)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def config_dict(cfg):
    return asdict(cfg)
