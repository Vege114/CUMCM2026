"""Recover a lost monthly prediction cache from its saved weights, without any fit call."""

import json
import os

import numpy as np

from .data import CHANNELS, HERE, ROOT, SEEDS, Data, month_origins

os.environ.setdefault("KERAS_HOME", str(ROOT / ".cache/keras"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def rebuild_prediction(stem, data, destination=None):
    # Importing registers the two custom Keras objects; this function never trains.
    import tensorflow as tf

    from . import train

    assert train.Branch is not None
    meta = json.loads(stem.with_suffix(".json").read_text())
    if not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("A GPU is required to reproduce the formal prediction caches")
    cutoff = meta["train_cutoff"]
    origins = np.arange(cutoff, month_origins(meta["month"])[-1]+1, 36)
    x, base, mean, scale = data.features(origins, cutoff)
    model = tf.keras.models.load_model(stem.with_suffix(".keras"))
    raw = np.concatenate([model(x[i:i+64], training=False).numpy()
                          for i in range(0, len(x), 64)]) * scale[CHANNELS] + base
    raw = np.maximum(raw, 0)
    predictions = raw.copy()
    for i, origin in enumerate(origins):
        ids = int(origin)+np.arange(144)
        history_days = min(28, int(origin)//144)
        history_ids = ids[None, :] - np.arange(1, history_days+1)[:, None]*144
        allowed = data.actual[np.maximum(history_ids, 0), 1].max(axis=0) > 0
        predictions[i, ~allowed, 1] = 0
        hourly = raw[i, 5::6, 3].copy()
        hourly[~(allowed[5::6] | (data.forecasts[int(origin)] > 0))] = 0
        predictions[i, :, 3] = data.interpolate_hourly(hourly, int(origin))
        raw[i, 5::6, 3] = hourly
    np.savez_compressed(destination or stem.with_suffix(".npz"), origins=origins,
        predictions=predictions.astype("float32"), hourly_pv=raw[:, 5::6, 3].astype("float32"),
        baseline=base, mean=mean, scale=scale, signature=meta["signature"], train_cutoff=cutoff,
        validation_count=meta["validation_origins"])


def recover_missing(run_id="exp002"):
    data = None
    recovered = []
    for month in range(2, 13):
        for seed in SEEDS:
            stem = HERE / "runs" / run_id / f"m{month:02d}_mlp_{seed}"
            if not stem.with_suffix(".npz").exists():
                data = data or Data()
                rebuild_prediction(stem, data)
                recovered.append(stem.name)
    return recovered
