"""Causal midnight arrays, seven-day sequences and fitted seasonal curves.

No future realized value enters features except for the explicitly marked
oracle_season mode. All forecast arrays have shape (144, 2) in kW.
"""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from experiments.problem2.exp003.data import DAYS, EPOCH, ROOT, STEPS, Data

HERE = Path(__file__).resolve().parent
OUT = ROOT / "data/results/exp004"
VARIANTS = ("no_season", "causal_season", "oracle_season")


def protocol():
    return json.loads((HERE / "protocol.json").read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_signature(data):
    sources = {str(p.relative_to(ROOT)): sha256(p) for p in
               [HERE / name for name in ("data.py", "train.py", "protocol.json")]}
    inherited = ROOT / "experiments/problem2/exp003/data.py"
    sources[str(inherited.relative_to(ROOT))] = sha256(inherited)
    payload = {"code": sources, "data": data.hashes}
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return signature, payload


def design(days):
    days = np.asarray(days, dtype=float).ravel()
    phase = 2 * np.pi * days / 365.25
    weekday = (days.astype(int) + 2) % 7
    return np.column_stack([np.ones(len(days)), np.sin(phase), np.cos(phase),
                            np.sin(2 * phase), np.cos(2 * phase),
                            *[(weekday == k).astype(float) for k in range(1, 7)]])


class Features:
    def __init__(self, data=None):
        self.data = Data() if data is None else data
        self.cfg = protocol()

    @lru_cache(maxsize=366)
    def seasonal_fit(self, cutoff_day):
        """Fit complete prefix; solve all 144 load/PV slot curves jointly."""
        if not 1 <= cutoff_day <= DAYS:
            raise ValueError("seasonal cutoff must be a complete observed day count")
        x = design(np.arange(cutoff_day))
        y = self.data.actual[:cutoff_day * STEPS].reshape(cutoff_day, -1)
        cfg = self.cfg["seasonality"]
        regularizer = np.diag([1e-8] + [cfg["harmonic_ridge"]] * 4
                              + [cfg["weekday_ridge"]] * 6)
        # Explicit contractions avoid spurious Accelerate/NumPy matmul FPE flags
        # on this host; nonfinite values remain a hard error rather than masked.
        gram = np.einsum("ni,nj->ij", x, x, optimize=False)
        rhs = np.einsum("ni,nj->ij", x, y, optimize=False)
        coef = np.linalg.solve(gram + regularizer, rhs)
        if not np.isfinite(coef).all():
            raise RuntimeError("Seasonal fit produced nonfinite coefficients")
        return coef

    def seasonal_shift(self, day, variant):
        if variant not in VARIANTS or not 7 <= day < DAYS:
            raise ValueError("invalid forecast variant or midnight day")
        cfg = self.cfg["seasonality"]
        if variant == "no_season" or day < cfg["minimum_days"]:
            return np.zeros((STEPS, 2), dtype=float)
        cutoff = DAYS if variant == "oracle_season" else day
        coef = self.seasonal_fit(cutoff)
        # Weekday effects are deliberately excluded from the seasonal shift.
        x = design([day, day - 7, day - 1])
        x[:, 5:] = 0
        curves = np.einsum("ni,ij->nj", x, coef, optimize=False).reshape(3, STEPS, 2)
        shift = np.column_stack((curves[0, :, 0] - curves[1, :, 0],
                                 curves[0, :, 1] - curves[2, :, 1]))
        scale = np.maximum(self.data.actual[:day * STEPS].std(axis=0), 1)
        bound = cfg["maximum_shift_scale"] * scale
        shrink = min(1, cutoff / cfg["shrink_days"])
        return np.clip(shift * shrink, -bound, bound)

    def base(self, day, variant):
        periodic = self.data.baseline(day * STEPS)
        return np.maximum(0, periodic + self.seasonal_shift(day, variant))

    def arrays(self, days, cutoff_day, variant):
        """Scaler cutoff is the training prefix, never the formal target day."""
        days = np.asarray(days, dtype=int)
        if (len(days) == 0 or days.min() < 7 or days.max() >= DAYS
                or not 7 <= cutoff_day <= days.max()):
            raise ValueError("invalid feature days/scaler cutoff")
        observed = self.data.actual[:cutoff_day * STEPS]
        mean = observed.mean(axis=0)
        scale = np.maximum(observed.std(axis=0), 1)
        sequences, contexts, bases, shifts, masks = [], [], [], [], []
        slotphase = 2 * np.pi * (np.arange(STEPS) + .5) / STEPS
        for day in days:
            origin = int(day) * STEPS
            history = self.data.actual[origin - 7 * STEPS:origin]
            # Hourly means retain chronological ordering of all seven days.
            sequences.append((history.reshape(168, 6, 2).mean(axis=1) - mean) / scale)
            base = self.base(int(day), variant)
            yesterday = self.data.baseline(origin, "yesterday")
            weekly = self.data.baseline(origin, "weekly")
            weekphase = 2 * np.pi * ((day + 2) % 7) / 7
            calendar = np.column_stack((np.sin(slotphase), np.cos(slotphase),
                                        np.full(STEPS, np.sin(weekphase)),
                                        np.full(STEPS, np.cos(weekphase))))
            contexts.append(np.stack([np.column_stack((
                (base[:, k] - mean[k]) / scale[k],
                (yesterday[:, k] - mean[k]) / scale[k],
                (weekly[:, k] - mean[k]) / scale[k],
                np.full(STEPS, (history[:, k].mean() - mean[k]) / scale[k]), calendar,
            )) for k in range(2)], axis=1))
            bases.append(base)
            shifts.append(self.seasonal_shift(int(day), variant))
            daylight = (self.data.actual[max(0, origin - 28 * STEPS):origin, 1]
                        .reshape(-1, STEPS) > 0).any(axis=0)
            # A fixed 20-minute margin permits upcoming sunrise/sunset shifts.
            mask = np.convolve(daylight.astype(int), np.ones(5), mode="same") > 0
            masks.append(mask)
        return {"sequence": np.asarray(sequences, dtype="float32"),
                "context": np.asarray(contexts, dtype="float32"),
                "base": np.asarray(bases), "seasonal_shift": np.asarray(shifts),
                "mask": np.asarray(masks), "mean": mean, "scale": scale}


def split_days(month):
    import pandas as pd

    if not 2 <= month <= 12:
        raise ValueError("formal months are February to December")
    asof = (pd.Timestamp(2025, month, 1) - EPOCH).days
    end = (pd.Timestamp(2025, month, 1) + pd.offsets.MonthEnd(0) - EPOCH).days + 1
    cutoff = asof - protocol()["training"]["validation_days"]
    return np.arange(7, cutoff), np.arange(cutoff, asof), np.arange(asof, end), cutoff


def age_weights(days, cutoff_day, half_life=90):
    days = np.asarray(days)
    if half_life <= 0 or np.any(days >= cutoff_day):
        raise ValueError("sample ages require strictly historical days and positive half-life")
    weights = 2.0 ** (-(cutoff_day - 1 - days) / half_life)
    return (weights / weights.mean()).astype("float32")
