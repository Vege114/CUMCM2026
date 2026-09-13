"""Question 2 data and causal features; no training-window policy is imposed.

Index ``o`` denotes an issue at the start of ten-minute interval ``o``.
Observed values have indices strictly below ``o``; targets are ``o:o+144``.
CSV times denote interval ends, including ``0:00+1`` for 24:00.
"""

import hashlib
import re
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
EPOCH = pd.Timestamp("2025-01-01")
STEPS = 144
DAYS = 365
TARGETS = ("load", "pv")
ALLOWED_INPUTS = ("附件1.csv", "附件2_小区负载.csv", "附件2_光伏发电实际功率.csv")


def _integer(value, name, minimum, maximum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return int(value)


def _origins(values, minimum=0):
    origins = np.asarray(values)
    if origins.ndim != 1 or (origins.size and origins.dtype.kind not in "iu"):
        raise ValueError("origins must be a one-dimensional sequence of integer interval indices")
    if origins.size and (origins.min() < minimum or origins.max() >= DAYS * STEPS):
        raise ValueError(f"origins must be between {minimum} and {DAYS * STEPS - 1}")
    return origins.astype(np.int64)


def _end_minutes(value):
    match = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?(\+1)?", str(value).strip())
    if not match:
        raise ValueError(f"invalid interval-end time: {value!r}")
    hour, minute, second = (int(part or 0) for part in match.groups()[:3])
    if second != 0 or minute >= 60 or hour >= 24:
        raise ValueError(f"invalid interval-end time: {value!r}")
    if match.group(4):
        if hour or minute:
            raise ValueError("the next-day endpoint must be 0:00+1")
        return 1440
    return 60 * hour + minute


def _validate_times(values, name):
    actual = np.asarray([_end_minutes(value) for value in values])
    if not np.array_equal(actual, np.arange(10, 1441, 10)):
        raise ValueError(f"{name}: expected ordered ten-minute interval ends from 00:10 to 0:00+1")


def _numeric(frame, name):
    try:
        values = frame.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name}: power and price cells must be numeric") from error
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{name}: power and price values must be finite and nonnegative")
    return values


class Data:
    """Read and hash exactly the three files permitted for Question 2.

    ``actual`` contains load/PV in kW, ``reference`` retains attachment 1's
    numeric price/load/PV columns, and ``fixed_price`` is in yuan/kWh.
    """

    def __init__(self, root=ROOT):
        self.root = Path(root)
        raw = self.root / "data/raw"
        self.hashes = {}
        frames = []
        for name in ALLOWED_INPUTS:
            content = (raw / name).read_bytes()
            self.hashes[name] = hashlib.sha256(content).hexdigest()
            frames.append(pd.read_csv(BytesIO(content)))

        reference, *histories = frames
        if reference.shape != (STEPS, 4):
            raise ValueError("附件1.csv: expected shape (144, 4)")
        if list(reference.columns) != ["时间", "电价", "小区负载", "光伏发电预测功率"]:
            raise ValueError("附件1.csv: unexpected time/price/load/PV column names or order")
        _validate_times(reference.iloc[:, 0], ALLOWED_INPUTS[0])
        self.reference = _numeric(reference.iloc[:, 1:], ALLOWED_INPUTS[0])
        self.fixed_price = self.reference[:, 0].copy()

        expected_dates = pd.date_range(EPOCH, periods=DAYS)
        arrays = []
        for name, frame in zip(ALLOWED_INPUTS[1:], histories):
            if frame.shape != (DAYS, STEPS + 1):
                raise ValueError(f"{name}: expected shape (365, 145)")
            _validate_times(frame.columns[1:], name)
            try:
                dates = pd.DatetimeIndex(pd.to_datetime(frame.iloc[:, 0], format="%Y-%m-%d", errors="raise"))
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name}: invalid dates") from error
            if not dates.equals(expected_dates):
                raise ValueError(f"{name}: expected every 2025 date exactly once in chronological order")
            arrays.append(_numeric(frame.iloc[:, 1:], name).ravel())
        self.actual = np.stack(arrays, axis=-1)

    def lag(self, origin, days):
        origin = _integer(origin, "origin", 0, DAYS * STEPS - 1)
        days = _integer(days, "days", 1, DAYS)
        targets = origin + np.arange(STEPS)
        output = self.reference[targets % STEPS][:, 1:3].copy()
        ids = targets - days * STEPS
        known = (ids >= 0) & (ids < origin)
        output[known] = self.actual[ids[known]]
        if days > 1:
            recent = targets - STEPS
            fallback = ~known & (recent >= 0) & (recent < origin)
            output[fallback] = self.actual[recent[fallback]]
        return output

    def baseline(self, origin, kind="periodic"):
        """Causal yesterday, weekly, or weekly-load/yesterday-PV predictions."""
        if kind not in ("periodic", "yesterday", "weekly"):
            raise ValueError(f"unknown baseline kind: {kind!r}")
        yesterday = self.lag(origin, 1)
        if kind == "yesterday":
            return yesterday
        weekly = self.lag(origin, 7)
        if kind == "periodic":
            weekly[:, 1] = yesterday[:, 1]
        return weekly

    def features(self, origins, cutoff):
        """Return features, bases, training means and scales for any issue slots.

        Shapes are ``(n,144,2,16)``, ``(n,144,2)``, ``(2,)``, ``(2,)``.
        ``cutoff`` is the caller's exclusive scaler-fit boundary. Historical
        training issues may precede it; callers must ensure it is no later
        than their model's deployment time. Features require at least one
        observed interval; issue zero is handled by the reference baseline.
        """
        origins = _origins(origins, minimum=1)
        if not origins.size:
            raise ValueError("features require at least one origin")
        cutoff = _integer(cutoff, "cutoff", 1, int(origins.max()))
        mean = self.actual[:cutoff].mean(axis=0)
        scale = np.maximum(self.actual[:cutoff].std(axis=0), 1e-6)
        features, bases = [], []
        for origin in origins:
            base = self.baseline(origin)
            yesterday, weekly = self.lag(origin, 1), self.lag(origin, 7)
            end = origin + np.arange(1, STEPS + 1)
            dayphase = end % STEPS / STEPS * 2 * np.pi
            weekphase = ((end // STEPS + 2) % 7) / 7 * 2 * np.pi
            issuephase = origin % STEPS / STEPS * 2 * np.pi
            calendar = np.column_stack((
                np.sin(dayphase), np.cos(dayphase), np.sin(weekphase), np.cos(weekphase),
                np.full(STEPS, np.sin(issuephase)), np.full(STEPS, np.cos(issuephase)),
                np.arange(1, STEPS + 1) / STEPS,
            ))
            branches = []
            for channel in range(2):
                history = self.actual[max(0, origin - 7 * STEPS):origin, channel]
                values = np.asarray((self.actual[origin - 1, channel], history[-36:].mean(),
                                     history[-STEPS:].mean(), history.mean()))
                stats = np.tile(np.r_[
                    (values - mean[channel]) / scale[channel],
                    history[-STEPS:].std() / scale[channel], history.std() / scale[channel],
                ], (STEPS, 1))
                branches.append(np.column_stack((
                    calendar, (yesterday[:, channel] - mean[channel]) / scale[channel],
                    (weekly[:, channel] - mean[channel]) / scale[channel], stats,
                    (base[:, channel] - mean[channel]) / scale[channel],
                )))
            features.append(np.stack(branches, axis=1))
            bases.append(base)
        return np.asarray(features, dtype="float32"), np.asarray(bases, dtype="float32"), mean, scale

    def labels(self, origins, cutoff=None):
        """Read complete target windows ending at or before exclusive ``cutoff``."""
        origins = _origins(origins)
        cutoff = DAYS * STEPS if cutoff is None else _integer(cutoff, "cutoff", 0, DAYS * STEPS)
        if origins.size and np.any(origins + STEPS > cutoff):
            raise ValueError("a complete 24-hour target window is unavailable at the label cutoff")
        ids = origins[:, None] + np.arange(STEPS)
        return self.actual[ids].astype("float32")


def midnight_origins(month=None, *, start=None, end=None):
    """Return daily 00:00 issue indices for a month or inclusive date range.

    With no arguments, return the formal evaluation period, February through
    December 2025. This helper does not prescribe training issue frequency.
    """
    if month is not None:
        month = _integer(month, "month", 1, 12)
        if start is not None or end is not None:
            raise ValueError("choose either month or start/end dates")
        first = pd.Timestamp(2025, month, 1)
        last = first + pd.offsets.MonthEnd(0)
    else:
        first = pd.Timestamp("2025-02-01" if start is None else start)
        last = pd.Timestamp("2025-12-31" if end is None else end)
    if any(value.tzinfo is not None or pd.isna(value) or value != value.normalize()
           for value in (first, last)):
        raise ValueError("start/end must be timezone-naive dates at midnight")
    if not EPOCH <= first <= last < EPOCH + pd.Timedelta(days=DAYS):
        raise ValueError("start/end must be an ordered inclusive range within 2025")
    return np.asarray((pd.date_range(first, last) - EPOCH).days, dtype=np.int64) * STEPS
