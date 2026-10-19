"""Domain features for swd, written from the schema and description only.

The ten inputs are anonymous ordinal risk-assessment factors, so these are generic but sensible
summaries of an overall risk profile.
"""
import numpy as np
import pandas as pd

COLS = ["In%d" % i for i in range(1, 11)]
# Number of levels per input, from the schema's unique counts.
LEVELS = {"In1": 3, "In2": 3, "In3": 4, "In4": 3, "In5": 4, "In6": 2, "In7": 3, "In8": 3, "In9": 3, "In10": 3}


def _scaled(df):
    # Map each ordinal input to [0, 1] using its number of levels so factors are comparable.
    out = {}
    for c in COLS:
        out[c] = (df[c].astype(float) - 1) / max(LEVELS[c] - 1, 1)
    return pd.DataFrame(out, index=df.index)


# Overall risk load: average of the level-normalized ordinal factors.
def mean_scaled(df):
    return _scaled(df).mean(axis=1)


# Raw sum of ordinal scores, as a social-work checklist total.
def raw_sum(df):
    return df[COLS].astype(float).sum(axis=1)


# Worst and best factor: a single severe factor can dominate a risk judgment.
def extremes(df):
    s = _scaled(df)
    return pd.DataFrame({"max_scaled": s.max(axis=1), "min_scaled": s.min(axis=1)}, index=df.index)


# Count of factors at their top and bottom level.
def counts_at_ends(df):
    s = _scaled(df)
    return pd.DataFrame({"n_top": (s >= 0.999).sum(axis=1).astype(float),
                         "n_bottom": (s <= 0.001).sum(axis=1).astype(float)}, index=df.index)


# Consistency of the profile: dispersion of the normalized factors.
def spread(df):
    return _scaled(df).std(axis=1)


PROPOSALS = [
    ("mean_scaled", mean_scaled),
    ("raw_sum", raw_sum),
    ("extremes", extremes),
    ("counts_at_ends", counts_at_ends),
    ("spread", spread),
]
