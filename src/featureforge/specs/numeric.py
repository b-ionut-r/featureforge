"""Same-row numeric features: pairwise arithmetic, row statistics over a column family, digit and
rounding residues, numbers inside coded strings, geographic distances and rotations, and the
label-free linear residual of one column on the others.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from featureforge.specs.base import Spec
from featureforge.utils.text import as_text


class Arith(Spec):
    def __init__(self, op: str, a: str, b: str):
        if op in ("add", "mul", "sub") and b < a:
            a, b = b, a  # symmetric (or sign-symmetric) ops: one canonical order
        super().__init__([a, b])
        self.op = op
        self.name = f"{a}__{op}__{b}"

    def transform(self, df, ctx):
        a = df[self.parents[0]].to_numpy(dtype=float)
        b = df[self.parents[1]].to_numpy(dtype=float)
        with np.errstate(all="ignore"):
            if self.op == "add":
                r = a + b
            elif self.op == "sub":
                r = a - b
            elif self.op == "mul":
                r = a * b
            elif self.op == "div":
                r = a / np.where(b == 0, np.nan, b)
            else:
                r = b / np.where(a == 0, np.nan, a)
        r[~np.isfinite(r)] = np.nan
        return r


class RowStat(Spec):
    """Row-wise statistic over a family of columns (e.g. one-hot blocks, repeated measurements)."""

    def __init__(self, cols: Sequence[str], stat: str, label: str):
        super().__init__(cols)
        self.stat = stat
        self.name = f"row_{stat}__{label}"

    def transform(self, df, ctx):
        # Row blocks: a wide family over a large frame (IEEE-CIS's 339 V columns over a
        # million rows) is gigabytes as one float64 matrix, plus the nan-function copies.
        # Each block is filled column by column: df[self.parents] copied the whole family over every
        # row for each block, and the four materialising threads did it at once (a 1,900-column
        # family: gigabytes per thread, killed at 12.5 GB in round 1). Column-major, as pandas
        # lays out a frame's matrix, so the row reductions sum in the same order.
        step = max(1, 2_000_000 // max(len(self.parents), 1))
        # Plain numeric columns are sliced as their own arrays (no copy) and cast on assignment, as
        # to_numpy(dtype=float) casts them: slicing each pandas column per block held the GIL for
        # most of a wide family's time (1,913 columns, 92,800 rows: 88 blocks x 1,900 slices a spec).
        cols = [df[c] for c in self.parents]
        cols = [s.to_numpy() if isinstance(s.dtype, np.dtype) and s.dtype.kind in "biuf" else s for s in cols]
        out = []
        for i in range(0, len(df), step):
            M = np.empty((min(step, len(df) - i), len(cols)), dtype=float, order="F")
            for j, s in enumerate(cols):
                M[:, j] = s[i:i + step] if isinstance(s, np.ndarray) else s.iloc[i:i + step].to_numpy(dtype=float)
            out.append(self._stat(M))
        return out[0] if len(out) == 1 else np.concatenate(out)

    def _stat(self, M):
        with np.errstate(all="ignore"):
            if self.stat == "sum":
                return np.nansum(M, axis=1)
            if self.stat == "mean":
                return np.nanmean(M, axis=1)
            if self.stat == "std":
                return np.nanstd(M, axis=1)
            if self.stat == "max":
                return np.nanmax(M, axis=1)
            if self.stat == "min":
                return np.nanmin(M, axis=1)
            if self.stat == "argmax":
                r = np.argmax(np.where(np.isnan(M), -np.inf, M), axis=1).astype(float)
                r[np.all(np.isnan(M), axis=1)] = np.nan
                return r
            if self.stat == "nonzero":
                return (np.nan_to_num(M) != 0).sum(axis=1).astype(float)
            if self.stat == "nan":
                return np.isnan(M).sum(axis=1).astype(float)
            # Ordered families (repeated measurements): trend over the running index.
            if self.stat == "slope":
                t = np.arange(M.shape[1], dtype=float)
                t -= t.mean()
                Mc = M - np.nanmean(M, axis=1, keepdims=True)
                return np.nansum(Mc * t, axis=1) / (t ** 2).sum()
            if self.stat == "delta":
                return M[:, 0] - M[:, -1]
            if self.stat == "npos":
                return (np.nan_to_num(M) > 0).sum(axis=1).astype(float)
        raise ValueError(self.stat)


class Digits(Spec):
    """Value-representation features: fractional part, last digits, rounding residue.

    Contest tables (prices, synthetic Playground data, sensor readings) often carry
    signal in how a value is written rather than in its magnitude.
    """

    def __init__(self, col: str, kind: str):
        super().__init__([col])
        self.kind = kind
        self.name = f"dig_{kind}__{col}"

    def transform(self, df, ctx):
        x = df[self.parents[0]].to_numpy(dtype=float)
        with np.errstate(all="ignore"):
            if self.kind == "frac":
                return x - np.floor(x)
            if self.kind == "mod10":
                return np.mod(np.round(x), 10)
            if self.kind == "mod100":
                return np.mod(np.round(x), 100)
            if self.kind == "frac100":   # cents
                return np.mod(np.round(x * 100), 100)
        raise ValueError(self.kind)


class StrNum(Spec):
    """The number written inside a coded string ("location 118", "feature 68"): codes
    issued in order put neighbouring numbers close, which the category alone hides."""
    # Its worth is in interactions (Telstra's location number: hand +0.02 log loss, no
    # residual gain on its own), so it is also offered to the gate on top of each feature
    # set. Exempt from the screen instead, it took a joint-model slot on West Nile's trap
    # code and the search stopped a round early (21 features, gate +7.2% -> 12, +2.9%).
    gate_addon = True

    def __init__(self, col: str):
        super().__init__([col])
        self.name = f"strnum__{col}"

    @staticmethod
    def parse(s: pd.Series) -> np.ndarray:
        return pd.to_numeric(as_text(s).str.extract(r"(\d+)\s*$")[0], errors="coerce").to_numpy(dtype=float)

    def transform(self, df, ctx):
        return self.parse(df[self.parents[0]])


class GeoPair(Spec):
    """Distance / bearing between two (lat, lon) points (haversine, km)."""

    def __init__(self, p: tuple, q: tuple, kind: str):
        super().__init__([p[0], p[1], q[0], q[1]])
        self.kind = kind
        self.name = f"geo_{kind}__{p[0]}__{q[0]}"

    def transform(self, df, ctx):
        la1, lo1, la2, lo2 = (np.radians(df[c].to_numpy(dtype=float)) for c in self.parents)
        dla, dlo = la2 - la1, lo2 - lo1
        with np.errstate(all="ignore"):
            if self.kind == "hav":
                a = np.sin(dla / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlo / 2) ** 2
                return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
            if self.kind == "manh":
                return 6371.0 * (np.abs(dla) + np.abs(dlo) * np.cos((la1 + la2) / 2))
            if self.kind == "bearing":
                yb = np.sin(dlo) * np.cos(la2)
                xb = np.cos(la1) * np.sin(la2) - np.sin(la1) * np.cos(la2) * np.cos(dlo)
                return np.degrees(np.arctan2(yb, xb))
        raise ValueError(self.kind)


class Rotate(Spec):
    """Coordinates rotated by a fixed angle: oblique boundaries become axis-aligned."""

    def __init__(self, a: str, b: str, deg: int):
        super().__init__([a, b])
        self.deg = deg
        self.name = f"rot{deg}__{a}__{b}"

    def transform(self, df, ctx):
        a = df[self.parents[0]].to_numpy(dtype=float)
        b = df[self.parents[1]].to_numpy(dtype=float)
        t = np.radians(self.deg)
        return a * np.cos(t) + b * np.sin(t)


class LinResid(Spec):
    """What the other numerics do not explain about one column (label-free).

    ``target`` is regressed (ridge) on the other ``cols`` and the residual is
    returned. Whole weight minus what the component weights predict, price
    minus what size and age predict: a linear combination over many columns
    that trees only approximate with many splits, and often the quantity that
    matters. ``log`` works on log1p of the non-negative columns
    (multiplicative structure: density, price per unit of size).
    """

    def __init__(self, cols: Sequence[str], target: str, log: bool = False):
        super().__init__([target] + [c for c in cols if c != target])
        self.log = log
        self.name = f"{'lresid' if log else 'resid'}__{target}"

    def _matrix(self, df, fit=False):
        M = df[self.parents].to_numpy(dtype=float)
        if fit:
            fin = np.where(np.isfinite(M), M, np.nan)
            self.log_ = (np.nanmin(fin, axis=0) >= 0) & self.log
        M = np.where(self.log_, np.log1p(np.where(self.log_, np.maximum(M, 0), 0)), M)
        if fit:
            self.lo_, self.hi_ = np.nanpercentile(M, 1, axis=0), np.nanpercentile(M, 99, axis=0)
        M = np.clip(M, self.lo_, self.hi_)
        if fit:
            self.med_ = np.nanmedian(M, axis=0)
        miss = ~np.isfinite(M[:, 0])
        M = np.where(np.isnan(M), self.med_, M)
        if fit:
            self.mu_, self.sd_ = M.mean(0), M.std(0) + 1e-12
        Z = (M - self.mu_) / self.sd_
        return Z, miss

    def fit(self, df, y, ctx):
        from sklearn.linear_model import Ridge
        Z, miss = self._matrix(df, fit=True)
        self.model_ = Ridge(alpha=1.0).fit(Z[~miss, 1:], Z[~miss, 0])
        return self

    def transform(self, df, ctx):
        Z, miss = self._matrix(df)
        r = Z[:, 0] - self.model_.predict(Z[:, 1:])
        r[miss] = np.nan
        return r
