"""Domain features for australian, written from the schema and description only.

Attribute names and values are anonymised (A1..A14), so these are mostly generic but sensible
transforms: logs of skewed continuous columns, ratios among them, and simple counts over binary flags.
"""
import numpy as np
import pandas as pd

CONT = ["A2", "A3", "A7", "A10", "A13", "A14"]
BIN = ["A1", "A8", "A9", "A11"]


def _f(df, c):
    return pd.to_numeric(df[c].astype(object), errors="coerce").astype(float)


def log_continuous(df):
    # continuous credit attributes (amounts/durations) are typically right-skewed; logs tame outliers
    return pd.DataFrame({"log_" + c: np.log1p(_f(df, c).clip(lower=0)) for c in ["A7", "A10", "A14", "A13"]})


def flag_count(df):
    # number of "true" binary flags: an applicant-level summary of the yes/no answers
    return pd.DataFrame({c: _f(df, c) for c in BIN}).sum(axis=1)


def a8_a9_combo(df):
    # A8/A9 are t/f flags often co-occurring; their joint pattern as a 0-3 code
    return _f(df, "A8") * 2 + _f(df, "A9")


def a10_given_a9(df):
    # A10 is a small count; interacting it with flag A9 (count only matters when the flag is on)
    return _f(df, "A10") * _f(df, "A9")


def ratio_a14_a13(df):
    # ratio of the two last continuous attributes (possible amount vs amount)
    return np.log1p(_f(df, "A14").clip(lower=0)) - np.log1p(_f(df, "A13").clip(lower=0))


def a7_per_a2(df):
    # A7 relative to A2 (e.g. duration relative to age-like quantity)
    return _f(df, "A7") / _f(df, "A2").replace(0, np.nan)


def cont_rank_mean(df):
    # mean of log continuous values: overall "magnitude" of the applicant's numbers
    return pd.DataFrame({c: np.log1p(_f(df, c).clip(lower=0)) for c in CONT}).mean(axis=1)


PROPOSALS = [("log_continuous", log_continuous), ("flag_count", flag_count), ("a8_a9_combo", a8_a9_combo),
             ("a10_given_a9", a10_given_a9), ("ratio_a14_a13", ratio_a14_a13), ("a7_per_a2", a7_per_a2),
             ("cont_rank_mean", cont_rank_mean)]
