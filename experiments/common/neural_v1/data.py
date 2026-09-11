"""Interval-end data and issue-time-only forecast features.

An origin is the number of completed ten-minute intervals since 2025-01-01.
The target at offset zero is the interval starting at that origin.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
TARGETS = ("load", "pv", "price", "pv_corrected")
SOURCE_CHANNELS = (0, 1, 2, 1)
STEPS = 144
EPOCH = pd.Timestamp("2025-01-01")


class Data:
    def __init__(self, root=ROOT):
        raw = Path(root) / "data/raw"
        names = ("附件2_小区负载.csv", "附件2_光伏发电实际功率.csv", "附件4.csv")
        frames = [pd.read_csv(raw / n) for n in names]
        expected = pd.date_range("2025-01-01", "2025-12-31")
        for frame in frames:
            assert pd.DatetimeIndex(pd.to_datetime(frame.iloc[:, 0])).equals(expected)
            assert frame.shape == (365, 145)
        self.actual = np.stack([f.iloc[:, 1:].to_numpy(float).ravel() for f in frames], -1)
        assert np.isfinite(self.actual).all() and (self.actual >= 0).all()
        self.reference = pd.read_csv(raw / "附件1.csv").iloc[:, 1:].to_numpy(float)
        assert self.reference.shape == (144, 3)
        self.fixed_price = self.reference[:, 0]
        f = pd.read_csv(raw / "附件3.csv")
        f.iloc[:, 0] = f.iloc[:, 0].ffill()
        self.forecasts = {}
        for row in f.itertuples(index=False, name=None):
            day = (pd.Timestamp(row[0]) - EPOCH).days
            hour = int(str(row[1]).split(":")[0])
            self.forecasts[day * 144 + hour * 6] = np.asarray(row[2:], float)
        assert len(self.forecasts) == 1460
        self.hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(raw.glob("*.csv"))}

    def issued_pv(self, origin):
        """Only this issue's forecast and the last completed observation are used."""
        anchor = self.actual[origin - 1, 1] if origin else 0.0
        return np.maximum(0, np.interp(np.arange(1, 145), np.arange(0, 145, 6),
                                      np.r_[anchor, self.forecasts[origin]]))

    def baseline(self, origin, days=1):
        ids = origin + np.arange(144) - days * 144
        fallback = np.column_stack((self.reference[:, 1], self.reference[:, 2],
                                    self.reference[:, 0]))
        out = fallback.copy()
        if days > 1 and origin >= 144:
            out = self.actual[origin - 144:origin].copy()
        known = (ids >= 0) & (ids < origin)
        out[known] = self.actual[ids[known]]
        return np.column_stack((out, self.issued_pv(origin)))

    def features(self, origins, cutoff, history_days=7, calendar=True):
        """Fit the scaler only on the training prefix ending at cutoff."""
        mean = self.actual[:cutoff].mean(0)
        scale = np.maximum(self.actual[:cutoff].std(0), 1e-6)
        hist, known, bases = [], [], []
        for o in origins:
            assert o >= history_days * 144
            h = self.actual[o - history_days * 144:o].reshape(history_days * 24, 6, 3).mean(1)
            hist.append((h - mean) / scale)
            target_end = o + np.arange(1, 145)
            phase = (target_end % 144) / 144 * 2 * np.pi
            # 2025-01-01 is Wednesday, index 2. Calendar is known in advance.
            week = ((target_end / 144 + 2) % 7) / 7 * 2 * np.pi
            cal = np.column_stack((np.sin(phase), np.cos(phase), np.sin(week),
                                   np.cos(week), np.arange(1, 145) / 144))
            if not calendar:
                cal[:, :4] = 0
            b = self.baseline(o)
            week_base = self.baseline(o, 7)
            features = [cal]
            for j, c in enumerate(SOURCE_CHANNELS):
                features.append(np.column_stack(((b[:, c] - mean[c]) / scale[c],
                                                 (week_base[:, c] - mean[c]) / scale[c],
                                                 (b[:, j] - mean[c]) / scale[c])))
            known.append(np.concatenate(features, -1))
            bases.append(b)
        return (np.asarray(hist, "float32"), np.asarray(known, "float32"),
                np.asarray(bases, "float32"), mean, scale)

    def labels(self, origins):
        indices = np.asarray(origins)[:, None] + np.arange(144)[None, :]
        assert indices.max() < len(self.actual)
        y = self.actual[indices]
        return np.concatenate((y, y[:, :, 1:2]), -1).astype("float32")


def month_origins(month):
    start = (pd.Timestamp(2025, month, 1) - EPOCH).days * 144
    end = (pd.Timestamp(2025, month, 1) + pd.offsets.MonthBegin(1) - EPOCH).days * 144
    return np.arange(start, end, 36, dtype=int)


def split_origins(month):
    asof = int(month_origins(month)[0])
    train_cutoff = asof - 7 * 144
    train = np.arange(7 * 144, train_cutoff - 144 + 1, 36, dtype=int)
    validation = np.arange(train_cutoff, asof - 144 + 1, 36, dtype=int)
    assert train[-1] + 144 <= train_cutoff
    assert validation[-1] + 144 <= asof
    return train, validation, train_cutoff
