"""Domain features for t03, written from the schema and description only.

Columns are anonymous positive continuous measurements (77 of them, looks like an expression/intensity profile),
so these are generic but sensible features: per-row level and spread, row-normalised profile, missingness.
"""
import numpy as np
import pandas as pd

_COLS = [f"f_{i}" for i in range(1, 78)]


def _mat(df):
    return df[_COLS].apply(pd.to_numeric, errors="coerce")


# Overall level per row: a sample-wide intensity/loading factor shared by all measurements.
def row_level(df):
    x = np.log(_mat(df).clip(lower=1e-6))
    return pd.DataFrame({
        "log_mean": x.mean(axis=1),
        "log_median": x.median(axis=1),
        "log_std": x.std(axis=1),
        "log_iqr": x.quantile(0.75, axis=1) - x.quantile(0.25, axis=1),
    })


# Row-normalised profile: remove the per-row loading factor so relative levels stand out.
def normalised_profile(df):
    x = np.log(_mat(df).clip(lower=1e-6))
    centred = x.sub(x.median(axis=1), axis=0)
    centred.columns = [c + "_rel" for c in centred.columns]
    return centred


# Missingness pattern: the columns with missing values may be missing for whole batches/groups.
def missingness(df):
    m = _mat(df).isna()
    sub = m[["f_69", "f_70", "f_72", "f_74", "f_75", "f_76", "f_24", "f_32", "f_28"]]
    return pd.DataFrame({
        "n_missing": m.sum(axis=1).astype(float),
        "missing_signature": (sub.astype(int) * (2 ** np.arange(sub.shape[1]))).sum(axis=1).astype(float),
    })


# Block summaries: the first 43 columns and the last 34 have visibly different scales/variability.
def block_summaries(df):
    x = np.log(_mat(df).clip(lower=1e-6))
    a = x[_COLS[:43]]
    b = x[_COLS[43:]]
    return pd.DataFrame({
        "blockA_mean": a.mean(axis=1),
        "blockB_mean": b.mean(axis=1),
        "blockA_minus_B": a.mean(axis=1) - b.mean(axis=1),
        "blockA_std": a.std(axis=1),
        "blockB_std": b.std(axis=1),
    })


# Low-rank projection: column-standardised log values projected on leading principal directions (label-free).
def pca_components(df):
    x = np.log(_mat(df).clip(lower=1e-6))
    x = x.fillna(x.median())
    z = (x - x.mean()) / x.std().replace(0, np.nan)
    z = z.fillna(0).values
    u, s, vt = np.linalg.svd(z, full_matrices=False)
    k = min(10, vt.shape[0])
    comps = z @ vt[:k].T
    return pd.DataFrame(comps, index=df.index, columns=[f"pc{i + 1}" for i in range(k)])


PROPOSALS = [
    ("row_level", row_level),
    ("normalised_profile", normalised_profile),
    ("missingness", missingness),
    ("block_summaries", block_summaries),
    ("pca_components", pca_components),
]
