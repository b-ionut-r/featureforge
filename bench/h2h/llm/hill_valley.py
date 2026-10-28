"""Domain features for hill_valley, written from the schema and description only.

V1..V100 are the ordered Y-values of a curve that contains either a bump (hill) or a dip (valley);
features describe series shape and are scale-free because raw magnitudes vary by orders of magnitude.
"""
import numpy as np
import pandas as pd


def _y(df):
    return df[["V%d" % i for i in range(1, 101)]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def _scale(y):
    s = np.nanstd(y, axis=1)
    s[s == 0] = np.nan
    return s


def peak_vs_trough(df):
    # hill: max sticks out above the baseline more than min sticks below it; valley the reverse
    y = _y(df)
    med = np.nanmedian(y, axis=1)
    up, dn = np.nanmax(y, 1) - med, med - np.nanmin(y, 1)
    tot = up + dn
    tot[tot == 0] = np.nan
    return pd.Series((up - dn) / tot, index=df.index)


def skewness(df):
    # a bump on flat terrain gives positive skew of the Y distribution, a dip negative skew
    y = _y(df)
    z = (y - np.nanmean(y, 1, keepdims=True)) / _scale(y)[:, None]
    return pd.Series(np.nanmean(z ** 3, axis=1), index=df.index)


def mean_minus_median(df):
    # mean pulled toward the bump/dip relative to the robust baseline, scale-normalised
    y = _y(df)
    return pd.Series((np.nanmean(y, 1) - np.nanmedian(y, 1)) / _scale(y), index=df.index)


def extreme_positions(df):
    # where the max and min sit (0-1); the feature (bump/dip) location vs the opposite extreme
    y = _y(df)
    return pd.DataFrame({"argmax": np.nanargmax(y, 1) / 99.0, "argmin": np.nanargmin(y, 1) / 99.0}, index=df.index)


def edge_vs_extreme(df):
    # compare the series ends (baseline) with the max/min: which extreme is far from the edge baseline
    y = _y(df)
    edge = np.concatenate([y[:, :10], y[:, -10:]], axis=1).mean(1)
    rng = np.nanmax(y, 1) - np.nanmin(y, 1)
    rng[rng == 0] = np.nan
    return pd.DataFrame({"max_above_edge": (np.nanmax(y, 1) - edge) / rng,
                         "edge_above_min": (edge - np.nanmin(y, 1)) / rng}, index=df.index)


def local_max_vs_window(df):
    # height of the extreme relative to a smoothed version: robust to noise spikes
    y = _y(df)
    k = 5
    ker = np.ones(k) / k
    sm = np.array([np.convolve(r, ker, mode="valid") for r in np.nan_to_num(y)])
    med = np.median(sm, 1)
    up, dn = sm.max(1) - med, med - sm.min(1)
    tot = up + dn
    tot[tot == 0] = np.nan
    return pd.Series((up - dn) / tot, index=df.index)


PROPOSALS = [("peak_vs_trough", peak_vs_trough), ("skewness", skewness), ("mean_minus_median", mean_minus_median),
             ("extreme_positions", extreme_positions), ("edge_vs_extreme", edge_vs_extreme),
             ("local_max_vs_window", local_max_vs_window)]
