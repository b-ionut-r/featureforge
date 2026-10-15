"""The ``Spec`` protocol every candidate feature implements.

A spec is fitted on training rows and transforms any rows. Target-dependent specs
(``target_dep = True``) also provide ``fit_transform_oof``: on training rows they return out-of-fold
values, so a row's own label never enters its feature. That is FeatureForge's leak rule.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from featureforge._frame import _code_cache_get, _code_cache_put, _codes, _fit_vocab


def _release(spec) -> None:
    """Drop a candidate's fitted state (attributes ending in ``_``). Thousands of candidates
    per round each holding lookup tables over a million rows (counts of a near-unique value,
    an entity's sorted event times) add up to gigabytes; rejected ones are never used again."""
    for a in [a for a in vars(spec) if a.endswith("_") and not a.startswith("_")]:
        delattr(spec, a)


class Spec:
    """A generated feature. ``transform`` returns an (n,) or (n, k) float array."""
    target_dep = False
    n_out = 1
    # Skips the one-column residual screen; the joint model and CV judge it instead.
    screen_exempt = False

    def __init__(self, parents: Sequence[str]):
        self.parents = list(parents)

    def fit(self, df, y, ctx):
        return self

    def transform(self, df, ctx) -> np.ndarray:
        raise NotImplementedError

    def _keys(self):
        return self.parents

    def _fit_codes(self, df) -> np.ndarray:
        hit = _code_cache_get(df, self._keys())
        if hit is not None:
            self.vocabs_ = hit[0]
            return hit[1].copy()
        self.vocabs_ = _fit_vocab(df, self._keys())
        codes = _codes(df, self._keys(), self.vocabs_)
        _code_cache_put(df, self._keys(), self.vocabs_, codes)
        return codes.copy()

    def _codes(self, df) -> np.ndarray:
        hit = _code_cache_get(df, self._keys())
        if hit is not None and hit[0] is self.vocabs_:
            return hit[1].copy()
        return _codes(df, self._keys(), self.vocabs_)

    def fit_transform_oof(self, df, y, ctx, folds) -> np.ndarray:
        self.fit(df, y, ctx)
        return self.transform(df, ctx)

    def out_names(self) -> List[str]:
        return [self.name] if self.n_out == 1 else [f"{self.name}_{k}" for k in range(self.n_out)]

    def __repr__(self):
        return self.name


class _OneHot:
    """``np.eye(n)[y]`` for TargetEnc, kept as the labels: it is only ever summed per key."""

    def __init__(self, y: np.ndarray, n: int):
        self.y, self.n = y, n

    def __getitem__(self, idx):
        return _OneHot(self.y[idx], self.n)

    def __len__(self):
        return len(self.y)
