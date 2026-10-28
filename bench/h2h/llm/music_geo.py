"""Domain features for music_geo, written from the schema and description only.

The audio columns V1..V116 are anonymous standardised MARSYAS features, so these are generic but
sensible summaries: block statistics over contiguous groups and label-free principal components.
Blocks V18-V29, V47-V58, V76-V87 and V105-V116 repeat one value each, so one column per block suffices.
"""
import numpy as np
import pandas as pd


def _cols(a, b):
    return ["V%d" % i for i in range(a, b + 1)]


_BLOCKS = {"b1": _cols(1, 17), "b2": _cols(30, 46), "b3": _cols(59, 75), "b4": _cols(88, 104)}


# Per-block mean, spread and energy of the varying features in each of the four segments.
def block_stats(df):
    out = {}
    for k, cols in _BLOCKS.items():
        x = df[cols].astype(float)
        out[k + "_mean"] = x.mean(axis=1)
        out[k + "_std"] = x.std(axis=1)
        out[k + "_l2"] = np.sqrt((x ** 2).sum(axis=1))
    return pd.DataFrame(out, index=df.index)


# The four repeated-constant columns, one representative each, and their spread.
def repeated_block_values(df):
    x = df[["V18", "V47", "V76", "V105"]].astype(float)
    out = x.copy()
    out.columns = ["rep1", "rep2", "rep3", "rep4"]
    out["rep_std"] = x.std(axis=1)
    return out


# Differences between parallel blocks (e.g. mean vs deviation statistics of the same descriptors).
def block_differences(df):
    a = df[_BLOCKS["b1"]].astype(float).values
    b = df[_BLOCKS["b2"]].astype(float).values
    c = df[_BLOCKS["b3"]].astype(float).values
    d = df[_BLOCKS["b4"]].astype(float).values
    return pd.DataFrame({
        "b1_minus_b3_mean": (a - c).mean(axis=1),
        "b2_minus_b4_mean": (b - d).mean(axis=1),
        "b1_b2_corr_proxy": (a * b).mean(axis=1),
    }, index=df.index)


# Label-free principal components of all audio features: dominant timbral directions.
def pca_components(df):
    cols = [c for c in df.columns if c.startswith("V")]
    x = df[cols].astype(float).fillna(0.0).values
    x = x - x.mean(axis=0)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    k = min(8, vt.shape[0])
    z = x @ vt[:k].T
    return pd.DataFrame(z, index=df.index, columns=["pc%d" % (i + 1) for i in range(k)])


# Shape of the first block as an ordered profile: slope and peak position.
def block1_shape(df):
    x = df[_BLOCKS["b1"]].astype(float).values
    t = np.arange(x.shape[1], dtype=float)
    t = t - t.mean()
    slope = (x * t).sum(axis=1) / (t ** 2).sum()
    return pd.DataFrame({"b1_slope": slope, "b1_argmax": x.argmax(axis=1).astype(float)}, index=df.index)


PROPOSALS = [
    ("block_stats", block_stats),
    ("repeated_block_values", repeated_block_values),
    ("block_differences", block_differences),
    ("pca_components", pca_components),
    ("block1_shape", block1_shape),
]
