"""Fixed-seed offline suite for later paired comparisons; no official UI calls."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260910, 20260911, 20260912])
    parser.add_argument("--prefix", default="offline")
    args = parser.parse_args()
    for problem in (3, 4):
        for seed in args.seeds:
            name = f"{args.prefix}_q{problem}_seed{seed}"
            if (ROOT/"runs"/name).exists():
                print(f"Preserving existing evidence: {name}")
                continue
            subprocess.run([sys.executable, str(ROOT/"run.py"), "--backend", "mock", "--problem", str(problem),
                            "--seed", str(seed), "--name", name], check=True)
    subprocess.run([sys.executable, str(ROOT/"evaluate_runs.py")], check=True)


if __name__ == "__main__":
    main()
