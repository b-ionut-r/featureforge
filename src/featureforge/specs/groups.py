"""Group and entity features: statistics of a numeric within its key's group, where a row sits in its
entity's history (gaps to the previous and next rows), in-group lags, leads and running sums, and
an entity's lagged target mean that ends one test horizon before the row.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from featureforge._frame import TE_SMOOTHING, _grouper
from featureforge.specs.base import Spec


class GroupStat(Spec):
    """Statistic of numeric ``num`` within groups of ``key``.

    ``key`` may be a tuple of columns: a composite group (``card`` x ``address``,
    the "user id" of fraud contests), whose statistics describe an entity no
    single column identifies. ``nunique`` counts distinct values of ``num``
    in the group.
    """

    def __init__(self, key, num: str, stat: str):
        keys = [key] if isinstance(key, str) else list(key)
        super().__init__(keys + [num])
        self.stat = stat
        self.name = f"grp_{stat}__{num}__by__{'+'.join(keys)}"

    def _keys(self):
        return self.parents[:-1]

    def fit(self, df, y, ctx):
        k = self._fit_codes(df)
        v = pd.Series(df[self.parents[-1]].to_numpy(dtype=float))
        cat, levels = _grouper(df, self._keys(), k)
        g = v.groupby(cat, observed=True)

        def by_key(a):
            return pd.Series(a.to_numpy(), index=levels)
        if self.stat == "nunique":
            self.a_ = by_key(g.nunique())
        elif self.stat in ("mean", "dev"):
            self.a_ = by_key(g.mean())
        elif self.stat == "std":
            self.a_ = by_key(g.std())
        elif self.stat == "min":
            self.a_ = by_key(g.min())
        elif self.stat == "max":
            self.a_ = by_key(g.max())
        elif self.stat == "z":
            self.a_ = by_key(g.mean())
            self.b_ = by_key(g.std())
        return self

    def transform(self, df, ctx):
        k = self._codes(df)
        a = self.a_.reindex(k).to_numpy(dtype=float)
        if self.stat == "dev":
            return df[self.parents[-1]].to_numpy(dtype=float) - a
        if self.stat == "z":
            b = self.b_.reindex(k).to_numpy(dtype=float)
            with np.errstate(all="ignore"):
                r = (df[self.parents[-1]].to_numpy(dtype=float) - a) / np.where(b > 0, b, np.nan)
            return r
        return a


class EntityLag(Spec):
    """Where a row sits in its entity's history: time since the entity's previous
    and until its next row, and how many earlier rows it has. Label-free; with
    unlabeled rows the history spans train and test, as in a contest."""

    def __init__(self, key, time: str, kind: str):
        keys = [key] if isinstance(key, str) else list(key)
        super().__init__(keys + [time])
        self.kind = kind
        self.name = f"lag_{kind}__{'+'.join(keys)}__{time}"

    @property
    def novelty_parents(self):
        # As for events: the entity key scopes the feature; it is not what it re-expresses.
        return [self.parents[-1]]

    def _keys(self):
        return self.parents[:-1]

    def _span(self):
        # Room for times well outside the fitted range (later test periods).
        R = self.t1_ - self.t0_ + 1.0
        return 6.0 * R, 2.5 * R - self.t0_

    def _pos(self, k, t):
        span, off = self._span()
        return k.astype(np.float64) * span + np.clip(t + off, 0.0, span - 1.0)

    def fit(self, df, y, ctx):
        k = self._fit_codes(df)
        t = df[self.parents[-1]].to_numpy(dtype=float)
        ok = np.isfinite(t) & (k >= 0)
        self.t0_, self.t1_ = float(np.nanmin(t)), float(np.nanmax(t))
        self.ref_ = np.sort(self._pos(k[ok], t[ok]))
        return self

    def transform(self, df, ctx):
        k = self._codes(df)
        t = df[self.parents[-1]].to_numpy(dtype=float)
        q = self._pos(k, np.nan_to_num(t, nan=self.t0_))
        span = self._span()[0]
        lo = np.searchsorted(self.ref_, q, side="left")    # first ref row at or after t
        hi = np.searchsorted(self.ref_, q, side="right")   # first ref row after t
        start = np.searchsorted(self.ref_, np.floor(q / span) * span, side="left")
        end = np.searchsorted(self.ref_, (np.floor(q / span) + 1) * span, side="left")
        ref = np.append(self.ref_, np.inf)
        if self.kind == "prev":
            r = np.where(lo > start, q - ref[np.maximum(lo - 1, 0)], np.nan)
        elif self.kind == "next":
            r = np.where(hi < end, ref[np.minimum(hi, len(self.ref_))] - q, np.nan)
        elif self.kind.startswith(("win", "fwd")):
            # Rows of the entity within ``w`` time units before (win) / after (fwd).
            w = float(self.kind[3:])
            if self.kind.startswith("win"):
                r = (lo - np.maximum(np.searchsorted(self.ref_, q - w, side="left"), start)).astype(float)
            else:
                r = (np.minimum(np.searchsorted(self.ref_, q + w, side="right"), end) - hi).astype(float)
        else:  # "nth": earlier rows of the entity
            r = (lo - start).astype(float)
        bad = ~np.isfinite(t) | (k < 0)
        r = r.astype(float)
        r[bad] = np.nan
        return r


class GroupSeq(Spec):
    """A row's place in its group's sequence: earlier and later values of a column (lags,
    leads, differences), its running sum and its time-weighted running sum (area), and the
    row's step within the group. Groups are whole sequences (breaths, sessions) whose rows
    the test holds in full; label-free, computed inside each group of the frame it sees.
    The per-breath lags and u_in cumsum of the Ventilator winners."""

    def __init__(self, key: str, order: str, col: Optional[str], kind: str):
        super().__init__([key, order] + ([col] if col else []))
        self.key, self.order, self.col, self.kind = key, order, col, kind
        self.name = f"seq_{kind}__{col or order}__{key}"

    @property
    def novelty_parents(self):
        # The group and its order scope the feature; it re-expresses the column.
        return [self.col or self.order]

    def transform(self, df, ctx):
        g = pd.factorize(df[self.key].to_numpy())[0]
        t = df[self.order].to_numpy(dtype=float)
        o = np.lexsort((t, g))
        gs, ts = g[o], t[o]
        start = np.r_[True, gs[1:] != gs[:-1]]
        first = np.maximum.accumulate(np.where(start, np.arange(len(gs)), 0))
        step = np.arange(len(gs)) - first
        k = self.kind
        if k == "step":
            r = step.astype(float)
        elif k == "dt":
            r = np.where(start, 0.0, np.diff(ts, prepend=np.nan))
        else:
            x = df[self.col].to_numpy(dtype=float)[o]
            if k.startswith(("lag", "lead", "diff")):
                m = int(k.lstrip("abcdefghijklmnopqrstuvwxyz"))
                sh = -m if k.startswith("lead") else m
                idx = np.arange(len(gs)) - sh
                ok = (idx >= 0) & (idx < len(gs))
                idx = np.clip(idx, 0, len(gs) - 1)
                ok &= gs[idx] == gs
                prev = np.where(ok, x[idx], np.nan)
                r = x - prev if k.startswith("diff") else prev
            elif k in ("cumsum", "area"):
                v = np.nan_to_num(x)
                if k == "area":
                    v = v * np.where(start, 0.0, np.diff(ts, prepend=np.nan))
                c = np.cumsum(v)
                r = c - (c - v)[first]
            elif k == "cummax":
                r = pd.Series(x).groupby(gs).cummax().to_numpy(dtype=float)
            else:
                raise ValueError(k)
        out = np.empty(len(gs))
        out[o] = r
        return out


class LaggedTargetMean(Spec):
    """Mean target of an entity's labelled rows in a trailing window that ends one
    test horizon before the row: ``[t - lag - window, t - lag)``. Training and test
    rows then see the same delay, so recent level shifts (a store's sales this
    quarter) carry into the test period. Uses only strictly earlier rows, so no
    out-of-fold scheme is needed."""
    target_dep = True
    time_safe = True

    def __init__(self, keys: Sequence[str], time: str, window: float, lag: float):
        super().__init__(list(keys) + [time])
        self.window, self.lag = float(window), float(lag)
        self.name = f"lagte__{'+'.join(keys)}__w{self.window:.6g}_l{self.lag:.6g}"

    @property
    def novelty_parents(self):
        return [self.parents[-1]]

    def _keys(self):
        return self.parents[:-1]

    def _pos(self, k, t):
        R = self.t1_ - self.t0_ + 1.0
        span, off = 6.0 * R, 2.5 * R - self.t0_
        return k.astype(np.float64) * span + np.clip(t + off, 0.0, span - 1.0), span

    def fit(self, df, y, ctx):
        k = self._fit_codes(df)
        t = df[self.parents[-1]].to_numpy(dtype=float)
        y = np.asarray(y, dtype=float)
        self.t0_, self.t1_ = float(np.nanmin(t)), float(np.nanmax(t))
        ok = np.isfinite(t) & (k >= 0) & np.isfinite(y)
        pos = self._pos(k[ok], t[ok])[0]
        o = np.argsort(pos, kind="stable")
        self.ref_ = pos[o]
        self.csum_ = np.concatenate([[0.0], np.cumsum(y[ok][o])])
        self.prior_ = float(np.mean(y[ok]))
        return self

    def transform(self, df, ctx):
        k = self._codes(df)
        t = df[self.parents[-1]].to_numpy(dtype=float)
        q, span = self._pos(k, np.nan_to_num(t, nan=self.t0_))
        block = np.floor(q / span) * span
        hi_q = np.maximum(q - self.lag, block)
        lo_q = np.maximum(q - self.lag - self.window, block)
        hi = np.searchsorted(self.ref_, hi_q, side="left")
        lo = np.searchsorted(self.ref_, lo_q, side="left")
        n = (hi - lo).astype(float)
        s = self.csum_[hi] - self.csum_[lo]
        r = (s + TE_SMOOTHING * self.prior_) / (n + TE_SMOOTHING)
        r[n == 0] = np.nan
        r[~np.isfinite(t) | (k < 0)] = np.nan
        return r

    def fit_transform_oof(self, df, y, ctx, folds):
        return self.fit(df, y, ctx).transform(df, ctx)
