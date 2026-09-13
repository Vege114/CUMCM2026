"""Issue-time feature construction; attachment 3 is loaded only on demand."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
EPOCH = pd.Timestamp("2025-01-01")
STEPS = 144
TARGETS = ("load", "pv", "price", "pv_corrected")
CHANNELS = np.array([0, 1, 2, 1])
SEEDS = (42, 2026, 3407)


def protocol():
    return json.loads((HERE / "protocol.json").read_text())


def signature(names, extra=None):
    digest = hashlib.sha256()
    for name in names:
        digest.update((HERE / name).read_bytes())
    digest.update(json.dumps(extra, sort_keys=True).encode())
    return digest.hexdigest()


class Data:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        raw = self.root / "data/raw"
        names = ("附件2_小区负载.csv", "附件2_光伏发电实际功率.csv", "附件4.csv")
        frames = [pd.read_csv(raw / name) for name in names]
        for frame in frames:
            assert frame.shape == (365, 145)
            assert pd.DatetimeIndex(pd.to_datetime(frame.iloc[:, 0])).equals(
                pd.date_range("2025-01-01", "2025-12-31"))
        self.actual = np.stack([f.iloc[:, 1:].to_numpy(float).ravel() for f in frames], -1)
        assert np.isfinite(self.actual).all() and (self.actual >= 0).all()
        self.reference = pd.read_csv(raw / "附件1.csv").iloc[:, 1:].to_numpy(float)
        self.fixed_price = self.reference[:, 0]
        self._forecasts = None
        self.hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(raw.glob("*.csv"))}

    @property
    def forecasts(self):
        if self._forecasts is None:
            frame = pd.read_csv(self.root / "data/raw/附件3.csv")
            frame.iloc[:, 0] = frame.iloc[:, 0].ffill()
            self._forecasts = {}
            for row in frame.itertuples(index=False, name=None):
                day = (pd.Timestamp(row[0]) - EPOCH).days
                hour = int(str(row[1]).split(":")[0])
                self._forecasts[day * 144 + hour * 6] = np.asarray(row[2:], float)
            assert len(self._forecasts) == 1460
        return self._forecasts

    def issued_points(self, origin):
        anchor = self.actual[origin - 1, 1] if origin else 0.0
        return np.maximum(0, np.interp(np.arange(1, 145), np.arange(0, 145, 6),
                                      np.r_[anchor, self.forecasts[origin]]))

    def integrate_points(self, points, origin):
        """kW interval averages; multiplying by 1/6 gives exact linear integral."""
        anchor = self.actual[origin - 1, 1] if origin else 0.0
        return (np.r_[anchor, points[:-1]] + points) / 2

    def interpolate_hourly(self, hourly, origin):
        anchor = self.actual[origin - 1, 1] if origin else 0.0
        points = np.interp(np.arange(1, 145), np.arange(0, 145, 6), np.r_[anchor, hourly])
        return self.integrate_points(points, origin)

    def lag(self, origin, days):
        target = origin + np.arange(144)
        reference = self.reference[target % 144][:, [1, 2, 0]].copy()
        ids = target - days * 144
        known = (ids >= 0) & (ids < origin)
        reference[known] = self.actual[ids[known]]
        if days > 1:
            recent = target - 144
            fallback = ~known & (recent >= 0) & (recent < origin)
            reference[fallback] = self.actual[recent[fallback]]
        return reference

    def baseline(self, origin, kind="periodic", issued=True, points=False):
        yesterday, weekly = self.lag(origin, 1), self.lag(origin, 7)
        if kind == "yesterday":
            output = yesterday.copy()
        elif kind == "weekly":
            output = weekly.copy()
        else:
            output = weekly.copy(); output[:, 1] = yesterday[:, 1]
        if issued:
            pv = self.issued_points(origin)
            output = np.column_stack((output, pv if points else self.integrate_points(pv, origin)))
        return output

    def features(self, origins, cutoff, issued=True):
        assert 0 < cutoff <= max(origins)
        mean = self.actual[:cutoff].mean(0)
        scale = np.maximum(self.actual[:cutoff].std(0), 1e-6)
        features, bases = [], []
        count = 4 if issued else 3
        for o in origins:
            assert o >= 7 * 144
            base = self.baseline(o, issued=issued, points=True)
            y, w = self.lag(o, 1), self.lag(o, 7)
            end = o + np.arange(1, 145)
            dayphase = end % 144 / 144 * 2 * np.pi
            weekphase = ((end // 144 + 2) % 7) / 7 * 2 * np.pi
            issuephase = o % 144 / 144 * 2 * np.pi
            cal = np.column_stack((np.sin(dayphase), np.cos(dayphase),
                                   np.sin(weekphase), np.cos(weekphase),
                                   np.full(144, np.sin(issuephase)),
                                   np.full(144, np.cos(issuephase)), np.arange(1,145)/144))
            branches = []
            for j, c in enumerate(CHANNELS[:count]):
                h = self.actual[max(0, o-168*6):o, c]
                values = [self.actual[o-1, c], h[-36:].mean(), h[-144:].mean(), h.mean()]
                stats = np.tile(np.r_[(np.array(values)-mean[c])/scale[c],
                                     h[-144:].std()/scale[c], h.std()/scale[c]], (144,1))
                branches.append(np.column_stack((cal, (y[:,c]-mean[c])/scale[c],
                    (w[:,c]-mean[c])/scale[c], stats, (base[:,j]-mean[c])/scale[c])))
            features.append(np.stack(branches, axis=1)); bases.append(base)
        return (np.asarray(features, "float32"), np.asarray(bases, "float32"), mean, scale)

    def labels(self, origins):
        ids = np.asarray(origins)[:, None] + np.arange(144)
        assert ids.max() < len(self.actual)
        return self.actual[ids][..., CHANNELS].astype("float32")


def month_origins(month):
    start = (pd.Timestamp(2025, month, 1) - EPOCH).days * 144
    end = (pd.Timestamp(2025, month, 1) + pd.offsets.MonthBegin(1) - EPOCH).days * 144
    return np.arange(start, end, 36, dtype=int)


def split_origins(month):
    asof = int(month_origins(month)[0]); cutoff = asof - 7 * 144
    train = np.arange(7*144, cutoff-144+1, 36, dtype=int)
    validation = np.arange(cutoff, asof-144+1, 36, dtype=int)
    return train, validation, cutoff
