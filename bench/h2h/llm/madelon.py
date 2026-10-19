"""Domain features for madelon, written from the schema and description only.

Columns V1..V500 are anonymous (synthetic data: 20 informative columns hidden among probes, order
randomised), so these are generic row-level summaries, not domain features."""
import numpy as np
import pandas as pd


def _X(df):
    return df[[f"V{i}" for i in range(1, 501)]].apply(pd.to_numeric, errors="coerce").astype(float)


def row_stats(df):
    # global row location/scale: values cluster around ~500, so per-row shifts may reflect cluster position
    X = _X(df)
    return pd.DataFrame({"row_mean": X.mean(axis=1), "row_std": X.std(axis=1), "row_median": X.median(axis=1)})


def row_extremes(df):
    # per-row spread at the tails: informative combinations push some values far from the centre
    X = _X(df)
    dev = (X - X.median(axis=1).values[:, None]).abs()
    return pd.DataFrame({"row_range": X.max(axis=1) - X.min(axis=1), "row_mad": dev.median(axis=1),
                         "n_far": (dev > 3 * dev.median(axis=1).replace(0, np.nan).values[:, None]).sum(axis=1)})


def column_zscore_norm(df):
    # distance from the dataset centre after per-column standardisation (radius in feature space)
    X = _X(df)
    Z = (X - X.mean()) / X.std().replace(0, np.nan)
    return pd.DataFrame({"z_norm": np.sqrt((Z ** 2).sum(axis=1)), "z_absmax": Z.abs().max(axis=1)})


def high_variance_group(df):
    # columns with the widest spread are likelier to be the informative ones (probes are narrower); summarise top 20
    X = _X(df)
    cols = X.std().sort_values(ascending=False).index[:20]
    Z = (X[cols] - X[cols].mean()) / X[cols].std().replace(0, np.nan)
    return pd.DataFrame({"hv_norm": np.sqrt((Z ** 2).sum(axis=1)), "hv_mean": Z.mean(axis=1)})


def block_means(df):
    # coarse block means over 5 groups of 100 columns
    X = _X(df)
    return pd.DataFrame({f"block{k}_mean": X.iloc[:, 100 * k:100 * (k + 1)].mean(axis=1) for k in range(5)})


PROPOSALS = [("row_stats", row_stats), ("row_extremes", row_extremes), ("column_zscore_norm", column_zscore_norm),
             ("high_variance_group", high_variance_group), ("block_means", block_means)]
