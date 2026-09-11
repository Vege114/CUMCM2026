"""Explicit checkpoint, code and upstream identities for portable result archives."""

import hashlib
import json

from .data import HERE, ROOT, SEEDS, Data, month_origins, protocol, signature


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prediction_content_digest(path):
    import numpy as np
    value = hashlib.sha256()
    with np.load(path) as arrays:
        for name in sorted(arrays.files):
            a = arrays[name]
            value.update(name.encode())
            value.update(str((a.dtype.str, a.shape)).encode())
            value.update(a.tobytes())
    return value.hexdigest()


def validate_upstream(run_id="exp002", replay=False, allow_missing_predictions=False):
    data = Data()
    expected = signature(("train.py", "data.py", "protocol.json"),
                         {"data": data.hashes, "epochs": protocol()["training"]["max_epochs"]})
    fingerprints = {}
    lock_path = HERE / "runs" / run_id / "upstream_fingerprints.json"
    old = json.loads(lock_path.read_text()) if lock_path.exists() else None
    for month in range(2, 13):
        for seed in SEEDS:
            path = HERE / "runs" / run_id / f"m{month:02d}_mlp_{seed}.json"
            if json.loads(path.read_text())["signature"] != expected:
                raise RuntimeError("Training code/data/config changed; use a new run-id rather than stale predictions")
            checkpoint = digest(path.with_suffix(".keras"))
            if path.with_suffix(".npz").exists():
                prediction = prediction_content_digest(path.with_suffix(".npz"))
            elif allow_missing_predictions and old is not None:
                prediction = old[path.stem]["prediction_content"]
            else:
                raise RuntimeError("Monthly predictions missing; recover through the predict stage")
            fingerprints[path.stem] = {"checkpoint": checkpoint, "prediction_content": prediction}
    if old is None:
        lock_path.write_text(json.dumps(fingerprints, indent=2))
    elif old != fingerprints:
        raise RuntimeError("Actual weights or cached forecast values changed; refusing to reuse upstream results")
    if replay:
        from .evaluate import evaluation_signature
        current = evaluation_signature(data, run_id)
        rows = json.loads((ROOT / "data/results" / run_id / "evaluation_manifest.json").read_text())
        if not rows or any(row["signature"] != current for row in rows):
            raise RuntimeError("Replay code or upstream checkpoint changed; regenerate before exporting/reporting")


def run(run_id="exp002"):
    validate_upstream(run_id)
    out = ROOT / "data/results" / run_id
    checkpoint_rows = []
    for month in range(2, 13):
        for seed in SEEDS:
            stem = HERE / "runs" / run_id / f"m{month:02d}_mlp_{seed}"
            metadata = json.loads(stem.with_suffix(".json").read_text())
            origins = month_origins(month)
            checkpoint_rows.append({"month": month, "seed": seed,
                "checkpoint": str(stem.with_suffix(".keras").relative_to(ROOT)),
                "sha256": digest(stem.with_suffix(".keras")),
                "prediction_cache_sha256": digest(stem.with_suffix(".npz")),
                "prediction_content_sha256": prediction_content_digest(stem.with_suffix(".npz")),
                "training_signature": metadata["signature"],
                "first_formal_origin": int(origins[0]), "last_formal_origin": int(origins[-1]),
                "information_asof": metadata["asof"], "train_cutoff": metadata["train_cutoff"]})
    archive = json.loads((out / "prediction_archive.json").read_text())
    archive.update(checkpoints=checkpoint_rows, origin_time="epoch + origin * 10 minutes",
                   target_interval="[origin+h, origin+h+1) * 10 minutes after epoch, h=0..143",
                   columns="seed_N[origin_index, future_interval, target_index]",
                   historical_replay={"source": "exp001/ensemble_predictions.npz",
                       "sha256": digest(ROOT / "data/results/exp001/ensemble_predictions.npz"),
                       "selection": "exp001/model_selection.csv; selected monthly official variant",
                       "pv_conversion": "adjacent saved ten-minute points and the observed anchor are trapezoidally integrated; no old training"})
    (out / "prediction_archive.json").write_text(json.dumps(archive, ensure_ascii=False, indent=2))
    (out / "protocol.json").write_text(json.dumps(protocol(), indent=2))
    (out / "source_hashes.json").write_text(json.dumps(
        {str(p.relative_to(ROOT)): digest(p) for p in sorted(HERE.glob("*.py"))}, indent=2))
    checks = {"formal_training_groups": len(checkpoint_rows), "months": 11, "seeds": list(SEEDS),
              "architecture": "one four-branch 4356-parameter MLP", "primary_seed": 42,
              "gpu_output_all": True, "zero_initialization_all": True,
              "parameters_updated_all": True, "save_reload_all": True}
    for row in checkpoint_rows:
        meta = json.loads((ROOT / row["checkpoint"]).with_suffix(".json").read_text())
        assert meta["parameters"] == 4356 and "GPU:0" in meta["device"]
        assert meta["zero_initial_output"] and meta["weights_updated"] and meta["save_reload_passed"]
    (out / "model_checks.json").write_text(json.dumps(checks, indent=2))


if __name__ == "__main__":
    run()
