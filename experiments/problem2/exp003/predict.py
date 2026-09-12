"""Audited Q2 forecast caches and residual mixing, without importing TensorFlow."""

import argparse
import hashlib
import json
from functools import lru_cache

import numpy as np
import pandas as pd

from .data import ALLOWED_INPUTS, EPOCH, HERE, ROOT, STEPS, Data, midnight_origins


def protocol():
    return json.loads((HERE / "protocol.json").read_text())


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_fingerprint(data):
    if set(data.hashes) != set(ALLOWED_INPUTS):
        raise ValueError("Q2 training fingerprints require exactly the three permitted inputs")
    codes = {name: file_hash(HERE / name) for name in ("data.py", "train.py", "predict.py", "protocol.json")}
    sources = {"code_hashes": codes, "data_hashes": data.hashes}
    signature = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    return signature, sources


def content_hash(arrays):
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(json.dumps([name, value.dtype.str, value.shape]).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def apply_residual(base, raw_residual, daylight_mask, alpha=(1., 1.)):
    """Mix in kW, then clip nonnegative and apply the historical PV night mask."""
    base = np.asarray(base, dtype=float)
    residual = np.asarray(raw_residual, dtype=float)
    mask = np.asarray(daylight_mask)
    alpha = np.asarray(alpha, dtype=float)
    if base.shape != residual.shape or base.shape[-1:] != (2,) or mask.shape != base.shape[:-1]:
        raise ValueError("base/residual must end in two channels with a matching daylight mask")
    if mask.dtype != np.bool_:
        raise ValueError("daylight_mask must be Boolean")
    if alpha.shape != (2,) or not np.isfinite(alpha).all() or (alpha < 0).any():
        raise ValueError("alpha must contain two finite nonnegative coefficients")
    if not np.isfinite(base).all() or not np.isfinite(residual).all() or (base < 0).any():
        raise ValueError("base must be finite/nonnegative and raw residuals finite")
    mixed = np.maximum(0., base + alpha * residual)
    mixed[..., 1] = np.where(mask, mixed[..., 1], 0.)
    if not np.isfinite(mixed).all():
        raise ValueError("residual mixing produced nonfinite predictions")
    return mixed


def historical_daylight(data, origins, history_days=None):
    """Use completed same-slot observations only; no future actual PV or issues."""
    days = protocol()["training"]["pv_mask_history_days"] if history_days is None else history_days
    masks = []
    for origin in np.asarray(origins, dtype=int):
        count = min(days, origin // STEPS)
        if count < 1:
            raise ValueError("daylight masks require at least one complete historical day")
        ids = origin + np.arange(STEPS)[None, :] - np.arange(1, count + 1)[:, None] * STEPS
        if ids.min() < 0 or ids.max() >= origin:
            raise ValueError("daylight mask reached unavailable observations")
        masks.append((data.actual[ids, 1] > 0).any(axis=0))
    return np.asarray(masks, dtype=bool)


class ForecastStore:
    def __init__(self, data, seed=42, run_id="exp003"):
        self.data, self.seed = data, int(seed)
        self.directory = HERE / "runs" / run_id
        self.signature, self.sources = training_fingerprint(data)

    @lru_cache(maxsize=11)
    def month(self, month):
        stem = self.directory / f"m{int(month):02d}_mlp_{self.seed}"
        meta = json.loads(stem.with_suffix(".json").read_text())
        if meta["signature"] != self.signature:
            raise RuntimeError(f"Checkpoint signature differs: {stem}; choose a new run-id")
        for suffix, key in ((".keras", "weights_sha256"), (".npz", "npz_sha256")):
            if file_hash(stem.with_suffix(suffix)) != meta[key]:
                raise RuntimeError(f"Checkpoint content hash differs: {stem}{suffix}")
        with np.load(stem.with_suffix(".npz"), allow_pickle=False) as archive:
            pack = {key: archive[key].copy() for key in archive.files}
        if content_hash(pack) != meta["prediction_content_sha256"]:
            raise RuntimeError(f"Prediction content hash differs: {stem}")
        asof = int(midnight_origins(int(month))[0])
        if meta["asof"] != asof or pack["signature"].item() != self.signature:
            raise RuntimeError(f"Checkpoint issuance metadata differs: {stem}")
        if not np.array_equal(pack["predictions"], apply_residual(
                pack["base"], pack["raw_residual"], pack["daylight_mask"]).astype("float32")):
            raise RuntimeError(f"Cached predictions do not match raw residuals: {stem}")
        pack["lookup"] = {int(origin): index for index, origin in enumerate(pack["origins"])}
        return pack

    def get(self, origin, alpha=(1., 1.), validation_month=None):
        if isinstance(origin, (bool, np.bool_)) or not isinstance(origin, (int, np.integer)):
            raise ValueError("origin must be an integer interval index")
        if origin < 0 or origin >= len(self.data.actual) or origin % STEPS:
            raise ValueError("Q2 cached forecasts are issued at 2025 midnight origins only")
        month = (EPOCH + pd.Timedelta(minutes=10 * int(origin))).month
        if month == 1 and validation_month is None:
            return self.data.baseline(origin)
        pack = self.month(validation_month or month)
        try:
            index = pack["lookup"][int(origin)]
        except KeyError as error:
            raise ValueError(f"origin {origin} is not available in the requested monthly checkpoint") from error
        return apply_residual(pack["base"][index], pack["raw_residual"][index], pack["daylight_mask"][index], alpha)


def run(run_id="exp003"):
    data, cfg = Data(), protocol()
    origins = midnight_origins()
    payload = {"origins": origins}
    metadata = []
    for seed in cfg["seed_list"]:
        store = ForecastStore(data, seed, run_id)
        residuals, bases, masks, predictions = [], [], [], []
        for month in range(2, 13):
            pack = store.month(month)
            indices = [pack["lookup"][int(origin)] for origin in midnight_origins(month)]
            residuals.append(pack["raw_residual"][indices])
            bases.append(pack["base"][indices])
            masks.append(pack["daylight_mask"][indices])
            predictions.append(pack["predictions"][indices])
            metadata.append(json.loads((store.directory / f"m{month:02d}_mlp_{seed}.json").read_text()))
        payload[f"seed_{seed}"] = np.concatenate(predictions)
        payload[f"residual_{seed}"] = np.concatenate(residuals)
        base, mask = np.concatenate(bases), np.concatenate(masks)
        if "base" in payload:
            np.testing.assert_array_equal(payload["base"], base)
            np.testing.assert_array_equal(payload["daylight_mask"], mask)
        else:
            payload.update(base=base, daylight_mask=mask)
    output = ROOT / "data/results" / run_id
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "predictions.npz", **payload)
    (output / "training_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    (output / "prediction_archive.json").write_text(json.dumps({
        "targets": ["load", "pv"], "origins": len(origins), "intervals": len(origins) * STEPS,
        "origin_unit_minutes": 10, "issue_hours": [0], "dimensions": ["origin", "future_interval", "target"],
        "primary_seed": cfg["primary_seed"], "averaged_seeds": False,
        "raw_prediction_alpha": [1, 1], "residual_unit": "kW",
        "mixing": "max(0,base+alpha*raw_residual), then historical PV night mask",
        "january_calibration": "m02 monthly caches; January defaults to causal periodic baseline",
        "signature": training_fingerprint(data)[0], "data_hashes": data.hashes,
        "archive_sha256": file_hash(output / "predictions.npz"),
        "prediction_content_sha256": content_hash(payload),
    }, ensure_ascii=False, indent=2))
    print(f"PREDICTIONS {len(origins)} midnight origins, {len(cfg['seed_list'])} seeds", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="exp003")
    run(**vars(parser.parse_args()))
