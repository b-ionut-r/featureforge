"""Date features: calendar fields of date columns, and time since (or until) an entity's last row
with a given event level.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from featureforge._frame import _as_str
from featureforge.specs.base import Spec
from featureforge.utils.text import as_text


_DAY = 86400.0
_DATE_PARTS = ("dow", "dom", "month", "year", "doy", "woy", "to_month_end")


def _to_days(s: pd.Series) -> np.ndarray:
    """Datetimes as float days since 1970-01-01 (NaN where missing)."""
    d = pd.to_datetime(s, errors="coerce")
    if getattr(d.dt, "tz", None) is not None:
        d = d.dt.tz_localize(None)
    v = d.to_numpy(dtype="datetime64[ns]").astype("int64").astype(float) / (_DAY * 1e9)
    v[d.isna().to_numpy()] = np.nan
    return v


def _is_date_like(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if not (s.dtype == object or pd.api.types.is_string_dtype(s)):
        return False
    sample = as_text(s.dropna().head(500))
    if len(sample) < 20 or not sample.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}").mean() > 0.95:
        return False
    return pd.to_datetime(sample, errors="coerce").notna().mean() > 0.95


class DatePart(Spec):
    """A calendar field of a date column (held internally as days since 1970)."""

    def __init__(self, col: str, part: str):
        super().__init__([col])
        self.part = part
        self.name = f"{part}__{col}"

    def transform(self, df, ctx):
        v = df[self.parents[0]].to_numpy(dtype=float)
        ok = np.isfinite(v)
        d = pd.to_datetime(np.where(ok, v, 0.0), unit="D")
        if self.part == "dow":
            r = d.dayofweek
        elif self.part == "dom":
            r = d.day
        elif self.part == "month":
            r = d.month
        elif self.part == "year":
            r = d.year
        elif self.part == "doy":
            r = d.dayofyear
        elif self.part == "woy":
            r = d.isocalendar().week.to_numpy()
        else:  # days to the end of the month
            r = d.days_in_month - d.day
        r = np.asarray(r, dtype=float)
        r[~ok] = np.nan
        return r


class EventRecency(Spec):
    """Time since the entity's last row where ``flag == level``, or until its next
    one (strictly before / after the row's own time). Rossmann-style calendar
    features: days since the store's last promotion, until the next holiday,
    since it last closed. Label-free; fitted over every row whose features are
    known, so test-period events count, as in the contest."""

    def __init__(self, keys: Sequence[str], time: str, flag: str, level, kind: str):
        super().__init__(list(keys) + [time, flag])
        self.level, self.kind = level, kind
        tag = "+".join(keys) if keys else "all"
        self.name = f"ev_{kind}__{flag}_{level}__{tag}"

    @property
    def novelty_parents(self):
        # The entity key only scopes the event: an entity's own level (a store's
        # usual sales) is no re-expression of when its promotions happen.
        return [self.parents[-1]]

    def _keys(self):
        return self.parents[:-2]

    def _key_codes(self, df, fit):
        if not self._keys():
            return np.zeros(len(df), dtype=np.int64)
        return self._fit_codes(df) if fit else self._codes(df)

    def _hit(self, df):
        f = df[self.parents[-1]]
        if isinstance(self.level, str):
            return (_as_str(f) == self.level).to_numpy()
        return (f.to_numpy(dtype=float) == self.level)

    def _pos(self, k, t):
        R = self.t1_ - self.t0_ + 1.0
        span, off = 6.0 * R, 2.5 * R - self.t0_
        return k.astype(np.float64) * span + np.clip(t + off, 0.0, span - 1.0), span

    def fit(self, df, y, ctx):
        k = self._key_codes(df, True)
        t = df[self.parents[-2]].to_numpy(dtype=float)
        self.t0_, self.t1_ = float(np.nanmin(t)), float(np.nanmax(t))
        ok = np.isfinite(t) & (k >= 0) & self._hit(df)
        self.ref_ = np.sort(self._pos(k[ok], t[ok])[0])
        return self

    def transform(self, df, ctx):
        k = self._key_codes(df, False)
        t = df[self.parents[-2]].to_numpy(dtype=float)
        q, span = self._pos(k, np.nan_to_num(t, nan=self.t0_))
        ref = np.append(self.ref_, np.inf)
        block = np.floor(q / span) * span
        if self.kind == "prev":
            lo = np.searchsorted(self.ref_, q, side="left")
            start = np.searchsorted(self.ref_, block, side="left")
            r = np.where(lo > start, q - ref[np.maximum(lo - 1, 0)], np.nan)
        else:
            hi = np.searchsorted(self.ref_, q, side="right")
            end = np.searchsorted(self.ref_, block + span, side="left")
            r = np.where(hi < end, ref[np.minimum(hi, len(self.ref_))] - q, np.nan)
        r = r.astype(float)
        r[~np.isfinite(t) | (k < 0)] = np.nan
        return r
