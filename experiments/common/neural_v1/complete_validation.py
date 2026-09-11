"""Predict the three incomplete-label validation origins without training on them."""

import argparse
import json

import numpy as np

from .data import ROOT, Data, month_origins
from .train import environment, predict, tf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="exp001")
    parser.add_argument("--months", nargs="+", type=int, default=list(range(2, 13)))
    args = parser.parse_args()
    environment(require_gpu=True)
    data = Data()
    directory = ROOT / "experiments/common/neural_v1/runs" / args.run_id
    for month in args.months:
        asof = int(month_origins(month)[0])
        origins = np.arange(asof - 108, asof, 36)
        for variant in ("mlp", "gru", "tcn"):
            h, k, b, _, scale = data.features(origins, asof - 7 * 144)
            for seed in (42, 2026, 3407):
                prefix = directory / f"m{month:02d}_{variant}_{seed}"
                meta = json.loads(prefix.with_suffix(".json").read_text())
                assert meta["asof"] == asof
                destination = str(prefix) + ".validation-tail.npz"
                if (directory / (prefix.name + ".validation-tail.npz")).exists():
                    with np.load(destination) as saved:
                        if saved['training_signature'].item() == meta['signature']:
                            np.testing.assert_array_equal(saved['origins'], origins)
                            continue
                tf.keras.backend.clear_session()
                model = tf.keras.models.load_model(prefix.with_suffix(".keras"))
                predictions = predict(model, h, k, b, scale)
                np.savez_compressed(destination,
                                    origins=origins, predictions=predictions,
                                    scored_until=asof, training_signature=meta["signature"])
        print(f"VALIDATION TAIL month={month} origins={origins.tolist()}", flush=True)


if __name__ == "__main__":
    main()
