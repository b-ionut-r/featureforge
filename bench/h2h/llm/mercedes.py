"""Domain features for mercedes, written from the schema and description only."""
import numpy as np
import pandas as pd

# Anonymised car configurations: X0-X8 categorical, the rest binary option flags.


def _bin_cols(df):
    return [c for c in df.columns if c not in ("X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8")]


def n_options(df):
    # more installed options -> more test modules to run -> longer test time
    b = df[_bin_cols(df)].apply(pd.to_numeric, errors="coerce")
    return b.sum(axis=1).astype(float)


def option_interactions(df):
    # option pairs that typically define test-bench programmes (e.g. X314/X315, X118/X127): combined flags
    g = lambda c: pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series(0.0, index=df.index)
    return pd.DataFrame({"x314_plus_x315": g("X314") + g("X315"),
                         "x118_x314_x315": g("X118") + g("X314") + g("X315"),
                         "x29_x127": g("X29") * 2 + g("X127"),
                         "x47_x232_x263": g("X47") + g("X232") + g("X263")}, index=df.index)


def options_vs_platform(df):
    # number of options relative to the platform (X0) average: extra equipment beyond the base model
    n = n_options(df)
    k = df["X0"].astype(object).fillna("__nan__").astype(str)
    return pd.DataFrame({"n_opt_minus_x0": n - n.groupby(k).transform("mean"),
                         "x0_count": k.groupby(k).transform("size").astype(float)}, index=df.index)


def binary_signature_frequency(df):
    # how many cars share the exact option set: identical configurations share test programmes
    cols = _bin_cols(df)
    sig = df[cols].astype(str).agg("".join, axis=1)
    return sig.groupby(sig).transform("size").astype(float)


def binary_pca_proxy(df):
    # weighted option score: each option weighted by its rarity (rare options add special tests)
    b = df[_bin_cols(df)].apply(pd.to_numeric, errors="coerce").fillna(0)
    p = b.mean(axis=0).clip(lower=1e-3, upper=1 - 1e-3)
    return pd.DataFrame({"rarity_score": (b * (-np.log(p))).sum(axis=1),
                         "common_score": (b * p).sum(axis=1)}, index=df.index)


def x0_x2_pair_count(df):
    # platform x configuration-line combinations: frequency of the (X0, X2) pairing
    k = df["X0"].astype(str) + "_" + df["X2"].astype(str)
    return k.groupby(k).transform("size").astype(float)


PROPOSALS = [
    ("option_interactions", option_interactions),
    ("options_vs_platform", options_vs_platform),
    ("n_options", n_options),
    ("binary_pca_proxy", binary_pca_proxy),
    ("binary_signature_frequency", binary_signature_frequency),
    ("x0_x2_pair_count", x0_x2_pair_count),
]
