"""Domain features for t08, written from the schema and description only.

Columns are anonymous small-integer codes; f_5 (range ~59-90) looks like age of older adults and f_3 (1/2) like a
sex code, but this is a guess. Features are generic: age bands, counts of codes, simple interactions and frequencies.
"""
import numpy as np
import pandas as pd


def _n(df, c):
    return pd.to_numeric(df[c].astype(object).astype(str).str.strip(), errors="coerce")


# Ordinal codes stored as text (f_6, f_7) converted to numbers, with a flag for non-numeric levels.
def ordinal_codes(df):
    a = _n(df, "f_6")
    b = _n(df, "f_7")
    return pd.DataFrame({
        "f6_num": a,
        "f7_num": b,
        "f6_nonnum": a.isna().astype(float),
        "f7_nonnum": b.isna().astype(float),
        "f7_positive": (b > 0).astype(float),
    })


# Age-like variable: decade band and squared term (effects of age are often nonlinear in older populations).
def age_shape(df):
    age = _n(df, "f_5")
    return pd.DataFrame({
        "age_band5": np.floor(age / 5),
        "age_sq": (age - age.mean()) ** 2,
        "age_ge_75": (age >= 75).astype(float),
    })


# Summed ordinal burden: total of the ordered code columns (like a severity / risk-factor count).
def code_sums(df):
    cols = ["f_1", "f_2", "f_4", "f_6", "f_7"]
    x = pd.DataFrame({c: _n(df, c) for c in cols})
    z = (x - x.mean()) / x.std().replace(0, np.nan)
    return pd.DataFrame({
        "code_sum": x.sum(axis=1),
        "code_z_sum": z.sum(axis=1),
        "n_nonzero_codes": (x > 0).sum(axis=1).astype(float),
    })


# Interactions with the age-like and sex-like columns.
def interactions(df):
    age = _n(df, "f_5")
    sex = _n(df, "f_3")
    return pd.DataFrame({
        "age_x_f6": age * _n(df, "f_6"),
        "age_x_f7": age * _n(df, "f_7"),
        "age_x_f1": age * _n(df, "f_1"),
        "sex_x_f2": sex * _n(df, "f_2"),
        "f1_x_f2": _n(df, "f_1") * _n(df, "f_2"),
    })


# Group frequency: how common each full code combination is (rare profiles may behave differently).
def combo_frequency(df):
    key = df[["f_1", "f_2", "f_3", "f_4", "f_6", "f_7"]].astype(str).agg("|".join, axis=1)
    return key.map(key.value_counts()).astype(float).rename("combo_count")


PROPOSALS = [
    ("ordinal_codes", ordinal_codes),
    ("age_shape", age_shape),
    ("code_sums", code_sums),
    ("interactions", interactions),
    ("combo_frequency", combo_frequency),
]
