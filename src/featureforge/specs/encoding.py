"""Encodings of keys and key interactions: frequency counts, smoothed out-of-fold target encodings
of keys, key pairs, key x binned numeric cells and 2-D numeric grids, column-family value counts,
and numbered slot families read as bags.
"""
from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from featureforge._frame import TE_SMOOTHING, _as_str, _codes, _fit_vocab
from featureforge.specs.base import Spec, _OneHot


class Count(Spec):
    def __init__(self, keys: Sequence[str]):
        super().__init__(keys)
        self.name = "count__" + "__".join(keys)

    def fit(self, df, y, ctx):
        k = self._fit_codes(df)
        self.table_ = pd.Series(k).value_counts()
        return self

    def transform(self, df, ctx):
        k = self._codes(df)
        return self.table_.reindex(k).fillna(0).to_numpy(dtype=float)


def _class_labels(Y) -> np.ndarray:
    """The class of each row of a one-hot target matrix, or of a ``_OneHot``."""
    return Y.y if isinstance(Y, _OneHot) else np.argmax(Y, axis=1)


class TargetEnc(Spec):
    """Smoothed target mean of a key (or key pair); out-of-fold on training rows."""
    target_dep = True

    def __init__(self, keys: Sequence[str], n_classes: int = 0):
        super().__init__(keys)
        self.name = "te__" + "__".join(keys)
        self.n_classes = n_classes
        self.n_out = n_classes if n_classes > 2 else 1

    def _targets(self, y):
        y = np.asarray(y, dtype=float)
        if self.n_out > 1:
            return _OneHot(y.astype(int), self.n_classes)
        return y[:, None]

    def _stats(self, k, Y):
        if isinstance(Y, _OneHot):
            # Per-key sums of one-hot rows are class counts: counted directly, the same
            # numbers without the rows x classes matrix (a hundred classes over 637k rows is
            # half a gigabyte per copy, and every fold took several).
            keys, inv = np.unique(k, return_inverse=True)
            n_k = len(keys)
            sums = np.bincount(inv * Y.n + Y.y, minlength=n_k * Y.n).reshape(n_k, Y.n).astype(float)
            prior = np.bincount(Y.y, minlength=Y.n) / len(Y.y)
            return (pd.DataFrame(sums, index=pd.Index(keys, name="_k")),
                    pd.Series(np.bincount(inv, minlength=n_k), index=pd.Index(keys, name="_k")), prior)
        frame = pd.DataFrame(Y)
        frame["_k"] = k
        g = frame.groupby("_k")
        return g.sum(), g.size(), Y.mean(axis=0)

    def _apply(self, k, sums, counts, prior):
        pos = sums.index.get_indexer(k)
        miss = pos < 0
        if len(sums) == 0:
            pos = np.zeros(len(k), dtype=np.intp)
            sums, counts = pd.DataFrame(np.zeros((1, sums.shape[1]))), pd.Series([0.0])
        s = sums.to_numpy(dtype=float)[pos]
        c = counts.to_numpy(dtype=float)[pos][:, None]
        s[miss] = np.nan
        c[miss] = np.nan
        s[np.isnan(s)] = 0.0
        c[np.isnan(c)] = 0.0
        s += TE_SMOOTHING * prior
        s /= c + TE_SMOOTHING
        return s[:, 0] if self.n_out == 1 else s

    def fit(self, df, y, ctx):
        k = self._fit_codes(df)
        self.sums_, self.counts_, self.prior_ = self._stats(k, self._targets(y))
        return self

    def transform(self, df, ctx):
        k = self._codes(df)
        return self._apply(k, self.sums_, self.counts_, self.prior_)

    def fit_transform_oof(self, df, y, ctx, folds):
        k = self._fit_codes(df)
        Y = self._targets(y)
        out = np.zeros((len(df), self.n_out))
        for tr, va in folds:
            r = self._apply(k[va], *self._stats(k[tr], Y[tr]))
            out[va] = r if r.ndim == 2 else r[:, None]
        self.fit(df, y, ctx)
        return out[:, 0] if self.n_out == 1 else out


class BinnedPairTE(TargetEnc):
    """Out-of-fold target map over a 2-D quantile grid of two numerics.

    Captures smooth two-way interactions (including rotated ones) that
    axis-aligned splits approximate with many leaves.
    """

    def __init__(self, a: str, b: str, n_classes: int = 0, n_bins: int = 16):
        super().__init__([a, b], n_classes)
        self.n_bins = n_bins
        self.name = f"te2d__{a}__{b}"

    def _bin(self, x, edges):
        finite = np.isfinite(x)
        return np.where(finite, np.searchsorted(edges, np.where(finite, x, 0.0), side="right"), len(edges) + 1)

    def _fit_codes(self, df):
        self.edges_ = []
        for c in self.parents:
            x = df[c].to_numpy(dtype=float)
            q = np.linspace(0, 1, self.n_bins + 1)[1:-1]
            self.edges_.append(np.unique(np.nanquantile(x[np.isfinite(x)], q)) if np.isfinite(x).any() else np.array([]))
        return self._codes(df)

    def _codes(self, df):
        a = self._bin(df[self.parents[0]].to_numpy(dtype=float), self.edges_[0])
        b = self._bin(df[self.parents[1]].to_numpy(dtype=float), self.edges_[1])
        return a.astype(np.int64) * (len(self.edges_[1]) + 2) + b


class _MixedCodes:
    """Joint code over keys and quantile-binned numerics (``binned`` names the latter)."""

    def _setup(self, binned: Sequence[str], n_bins: int):
        self.binned = list(binned)
        self.n_bins = n_bins

    def _fit_codes(self, df):
        self.edges_ = {}
        q = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        for c in self.binned:
            x = df[c].to_numpy(dtype=float)
            fin = np.isfinite(x)
            self.edges_[c] = np.unique(np.nanquantile(x[fin], q)) if fin.any() else np.array([])
        keys = [c for c in self.parents if c not in self.edges_]
        self.vocabs_ = _fit_vocab(df, keys)
        return self._codes(df)

    def _codes(self, df):
        out = np.zeros(len(df), dtype=np.int64)
        vocabs = iter(self.vocabs_)
        for c in self.parents:
            if c in self.edges_:
                e = self.edges_[c]
                x = df[c].to_numpy(dtype=float)
                fin = np.isfinite(x)
                idx = np.where(fin, np.searchsorted(e, np.where(fin, x, 0.0), side="right"), len(e) + 1)
                out = out * (len(e) + 2) + idx
            else:
                vocab = next(vocabs)
                out = out * (len(vocab) + 1) + _codes(df, [c], [vocab])
        return out


class MixedTE(_MixedCodes, TargetEnc):
    """Out-of-fold target encoding of a key/numeric interaction cell.

    Numerics are quantile-binned, so ``(city, income_bin)`` or
    ``(age_bin, hours_bin, sex)`` become categorical cells whose target rate
    is learned directly instead of through many tree splits.
    """

    def __init__(self, cols: Sequence[str], binned: Sequence[str], n_classes: int = 0, n_bins: int = 8):
        TargetEnc.__init__(self, cols, n_classes)
        self._setup(binned, n_bins)
        self.name = "tex__" + "__".join(f"{c}~{n_bins}" if c in binned else c for c in cols)


class MixedCount(_MixedCodes, Count):
    """Frequency of a key/binned-numeric interaction cell."""

    def __init__(self, cols: Sequence[str], binned: Sequence[str], n_bins: int = 8):
        Count.__init__(self, cols)
        self._setup(binned, n_bins)
        self.name = "cntx__" + "__".join(f"{c}~{n_bins}" if c in binned else c for c in cols)


def _value_table(s: pd.Series, ones: bool = True):
    """``s.value_counts()`` held as sorted arrays: the repeated values with their counts, and
    (``ones``) the values seen once. A mapped Series keeps a hash table on its index, and a
    family of a thousand near-unique columns holds gigabytes of them; repeats are all a mask
    needs. Non-numeric columns keep the Series."""
    vc = s.value_counts()
    if vc.index.dtype.kind not in "iuf" or isinstance(vc.index.dtype, pd.api.extensions.ExtensionDtype):
        return vc
    v, k = vc.index.to_numpy(), vc.to_numpy()
    rep = k > 1
    o = np.argsort(v[rep], kind="stable")
    one = np.sort(v[~rep]) if ones else None
    return v[rep][o], k[rep][o], one


def _sorted_hits(table: np.ndarray, x: np.ndarray):
    """Positions of x in the sorted ``table``, and where they match exactly."""
    if len(table) == 0:
        return np.zeros(len(x), dtype=np.intp), np.zeros(len(x), dtype=bool)
    i = np.minimum(np.searchsorted(table, x), len(table) - 1)
    return i, table[i] == x


def _map_counts(s: pd.Series, vc) -> np.ndarray:
    """``s.map(vc).to_numpy(dtype=float)`` for a table from ``_value_table`` (NaN where a value
    was not counted; seen-once values need ``ones``)."""
    if isinstance(vc, pd.Series):
        return s.map(vc).to_numpy(dtype=float)
    v, k, one = vc
    if s.dtype.kind not in "iuf" or isinstance(s.dtype, pd.api.extensions.ExtensionDtype):
        full = pd.Series(np.r_[k, np.ones(0 if one is None else len(one), k.dtype)],
                         index=np.r_[v, [] if one is None else one])
        return s.map(full).to_numpy(dtype=float)
    x = s.to_numpy()
    cnt = np.full(len(x), np.nan)
    i, hit = _sorted_hits(v, x)
    cnt[hit] = k[i[hit]]
    if one is not None:
        _, hit1 = _sorted_hits(one, x)
        cnt[hit1] = 1.0
    return cnt


class FamilyCount(Spec):
    """Value frequencies of every column of a homogeneous family, as one block.

    ``kind="count"``: how often each row's value occurs in its column (over every
    row whose features are known). ``kind="mask"``: the value itself where it
    repeats, NaN where it is unique. Santander-style anonymised tables hide
    signal in whether a value is shared; each column alone adds little, the
    whole block a lot, so the block is screened by summing per-column novel
    gains (``paired``).
    """
    paired = True

    def __init__(self, cols: Sequence[str], kind: str, label: str):
        super().__init__(cols)
        self.kind = kind
        self.n_out = len(cols)
        self.name = f"fam{kind}__{label}"

    def out_names(self):
        return [f"fam{self.kind}__{c}" for c in self.parents]

    def fit(self, df, y, ctx):
        self.vc_ = [_value_table(df[c], ones=self.kind == "count") for c in self.parents]
        return self

    def transform(self, df, ctx):
        out = np.empty((len(df), self.n_out), dtype=np.float32)
        for j, (c, vc) in enumerate(zip(self.parents, self.vc_)):
            cnt = _map_counts(df[c], vc)
            if self.kind == "count":
                out[:, j] = np.nan_to_num(cnt, nan=0.0)
            else:
                x = df[c].to_numpy(dtype=float)
                out[:, j] = np.where(cnt > 1, x, np.nan)
        return out


def _slot_values(df, cols) -> np.ndarray:
    v = np.column_stack([_as_str(df[c]).to_numpy() for c in cols])
    return np.where(np.isin(v, ["__NA__", "nan", "None", "NA", ""]), None, v)


class SlotBag(Spec):
    """A multi-valued field stored in numbered slots, as a bag: number of filled slots,
    and how many times (or how much, with a paired quantity family) each frequent
    value occurs across the slots, whatever slot it sits in. Label-free."""

    def __init__(self, cols: Sequence[str], weights: Optional[Sequence[str]], label: str, top: int = 16):
        super().__init__(list(cols) + list(weights or []))
        self.cols, self.weights, self.top = list(cols), list(weights or []), top
        self.name = f"slotbag__{label}"
        self.n_out = 1 + top

    def out_names(self):
        return [f"{self.name}_n"] + [f"{self.name}_{k}" for k in range(self.n_out - 1)]

    def fit(self, df, y, ctx):
        extra = getattr(ctx, "extra_rows", None)
        ref = df if extra is None else pd.concat([df[self.parents], extra[self.parents]], ignore_index=True)
        v = _slot_values(ref, self.cols).ravel()
        vc = pd.Series(v[v != None]).value_counts()  # noqa: E711
        self.levels_ = list(vc.index[:self.top])  # fewer than ``top``: the rest stay zero
        return self

    def transform(self, df, ctx):
        V = _slot_values(df, self.cols)
        W = (np.column_stack([df[c].to_numpy(dtype=float) for c in self.weights]) if self.weights
             else np.ones(V.shape))
        W = np.nan_to_num(W, nan=1.0)
        out = [(V != None).sum(1).astype(float)]  # noqa: E711
        for lv in self.levels_:
            out.append(((V == lv) * W).sum(1))
        out += [np.zeros(len(V))] * (self.n_out - len(out))
        return np.column_stack(out).astype(np.float32)


class SlotTE(Spec):
    """Target maps of the values of a slot family, learned across all slots (each row's
    target counts once for every value it holds), then summarised per row: mean, max,
    min and (quantity-)weighted sum over its filled slots. Out-of-fold on training rows."""
    target_dep = True
    n_out = 4

    def __init__(self, cols: Sequence[str], weights: Optional[Sequence[str]], label: str):
        super().__init__(list(cols) + list(weights or []))
        self.cols, self.weights = list(cols), list(weights or [])
        self.name = f"slotte__{label}"

    def _table(self, V, y):
        rows = np.repeat(np.arange(len(V)), V.shape[1])
        v = V.ravel()
        ok = v != None  # noqa: E711
        f = pd.DataFrame({"v": v[ok], "y": np.asarray(y, dtype=float)[rows[ok]]})
        g = f.groupby("v")["y"].agg(["sum", "size"])
        prior = float(np.mean(y))
        return (g["sum"] + TE_SMOOTHING * prior) / (g["size"] + TE_SMOOTHING), prior

    def _apply(self, V, W, tab, prior):
        E = tab.reindex(V.ravel()).to_numpy(dtype=float).reshape(V.shape)
        E = np.where(V == None, np.nan, np.where(np.isnan(E), prior, E))  # noqa: E711
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.column_stack([np.nanmean(E, 1), np.nanmax(E, 1), np.nanmin(E, 1),
                                    np.nansum(np.nan_to_num(E - prior) * W, 1)])

    def _vw(self, df):
        V = _slot_values(df, self.cols)
        W = (np.nan_to_num(np.column_stack([df[c].to_numpy(dtype=float) for c in self.weights]), nan=1.0)
             if self.weights else np.ones(V.shape))
        return V, W

    def fit(self, df, y, ctx):
        V, _ = self._vw(df)
        self.tab_, self.prior_ = self._table(V, y)
        return self

    def transform(self, df, ctx):
        V, W = self._vw(df)
        return self._apply(V, W, self.tab_, self.prior_)

    def fit_transform_oof(self, df, y, ctx, folds):
        V, W = self._vw(df)
        y = np.asarray(y, dtype=float)
        out = np.full((len(df), self.n_out), np.nan)
        for tr, va in folds:
            out[va] = self._apply(V[va], W[va], *self._table(V[tr], y[tr]))
        self.fit(df, y, ctx)
        return out


class KeyBinTE(BinnedPairTE):
    """Out-of-fold target map over (key level x quantile bin of a numeric).

    The categorical-by-numeric interaction ("price band within this model",
    "age band within this country") that a tree needs one split per level for.
    """

    def __init__(self, key: str, num: str, n_classes: int = 0, n_bins: int = 8):
        TargetEnc.__init__(self, [key, num], n_classes)
        self.n_bins = n_bins
        self.name = f"tekb__{key}__{num}"

    def _fit_codes(self, df):
        self.vocabs_ = _fit_vocab(df, self.parents[:1])
        x = df[self.parents[1]].to_numpy(dtype=float)
        q = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.edges_ = np.unique(np.nanquantile(x[np.isfinite(x)], q)) if np.isfinite(x).any() else np.array([])
        return self._codes(df)

    def _codes(self, df):
        k = _codes(df, self.parents[:1], self.vocabs_)
        b = self._bin(df[self.parents[1]].to_numpy(dtype=float), self.edges_)
        return k * (len(self.edges_) + 2) + b.astype(np.int64)
