"""Re-evaluate saved runs and build an inspectable version-comparison index."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from Benchmark.evaluate import evaluate, read_jsonl


def main():
    index = []
    for path in sorted((ROOT/"runs").glob("*/metadata.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        metrics = evaluate(read_jsonl(path.parent/"events.jsonl"), meta)
        (path.parent/"metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        index.append({"run_id": meta["run_id"], "problem": meta["problem"], "source": meta["source"],
                      "mode": meta["mode"], "algorithm": meta["algorithm"], "seed": meta.get("seed"),
                      **metrics["official_metrics"], "validation": metrics["validation"],
                      "metrics_path": (path.parent/"metrics.json").relative_to(ROOT).as_posix()})
    (ROOT/"benchmark_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"runs": len(index), "valid": sum(r["validation"]["valid_run"] for r in index),
                      "all_cleared": sum(r["validation"]["all_cleared_verified"] for r in index)}))


if __name__ == "__main__":
    main()
