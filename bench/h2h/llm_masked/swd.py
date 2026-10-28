"""Domain features for data, written from the schema and description only.

Columns are anonymous small ordinal codes (2-4 levels each), so these are generic features: sums, counts of
extreme levels and normalised totals across the 10 columns.
"""
import numpy as np
import pandas as pd

_COLS = ["f_%d" % i for i in range(1, 11)]


def _x(df):
    return df[_COLS].apply(pd.to_numeric, errors="coerce").astype(float)


def level_sums(df):
    # plain additive score across all ordinal items
    x = _x(df)
    return pd.DataFrame({"sum": x.sum(axis=1), "std": x.std(axis=1)})


def normalised_sum(df):
    # each item rescaled to [0,1] by its own level range, then averaged (items have different numbers of levels)
    x = _x(df)
    mn, mx = x.min(), x.max()
    z = (x - mn) / (mx - mn).replace(0, np.nan)
    return pd.DataFrame({"norm_mean": z.mean(axis=1), "norm_min": z.min(axis=1), "norm_max": z.max(axis=1)})


def extreme_counts(df):
    # how many items sit at their lowest / highest level
    x = _x(df)
    return pd.DataFrame({
        "n_at_min": x.eq(x.min()).sum(axis=1).astype(float),
        "n_at_max": x.eq(x.max()).sum(axis=1).astype(float),
        "n_ones": x.eq(1).sum(axis=1).astype(float),
    })


def pattern_rarity(df):
    # frequency of each item's level in the table, averaged: rare combinations of levels
    x = _x(df)
    f = pd.DataFrame({c: x[c].map(x[c].value_counts(normalize=True)) for c in _COLS})
    return pd.DataFrame({"mean_level_freq": f.mean(axis=1), "min_level_freq": f.min(axis=1)})


def half_contrasts(df):
    # contrast between first and second half of the items, and adjacent-pair products
    x = _x(df)
    out = {"first_minus_second": x[_COLS[:5]].sum(axis=1) - x[_COLS[5:]].sum(axis=1)}
    for i in range(0, 10, 2):
        out["prod_%d_%d" % (i + 1, i + 2)] = x[_COLS[i]] * x[_COLS[i + 1]]
    return pd.DataFrame(out)


PROPOSALS = [
    ("level_sums", level_sums),
    ("normalised_sum", normalised_sum),
    ("extreme_counts", extreme_counts),
    ("pattern_rarity", pattern_rarity),
    ("half_contrasts", half_contrasts),
]
