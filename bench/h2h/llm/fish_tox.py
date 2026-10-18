"""Domain features for fish_tox, written from the schema and description only."""
import numpy as np
import pandas as pd


# Baseline narcosis: toxicity rises with lipophilicity, with curvature (bioavailability cutoff).
# OUTSIDE KNOWLEDGE: parabolic logP dependence of narcotic toxicity (Hansch-type QSAR).
def mlogp_poly(df):
    m = df["MLOGP"].astype(float)
    return pd.DataFrame({"mlogp_sq": m ** 2, "mlogp_pos": m.clip(lower=0)}, index=df.index)


# Unsaturation: sp2 carbon counts (=CH- and =C<) may indicate reactive (e.g. Michael acceptor) groups.
def unsaturated_carbons(df):
    return df["NdsCH"].astype(float) + df["NdssC"].astype(float)


# Any unsaturated carbon present.
def has_unsaturation(df):
    return ((df["NdsCH"] + df["NdssC"]) > 0).astype(float)


# Lipophilicity times electronic/polarity descriptor: reactive lipophilic chemicals are most toxic.
def mlogp_x_sm1(df):
    return df["MLOGP"] * df["SM1_Dz"]


# Lipophilicity relative to molecular complexity (information index).
def mlogp_per_cic(df):
    return df["MLOGP"] / df["CIC0"].replace(0, np.nan)


# Ionisation-related autocorrelation interacting with polarity descriptor.
def gats_x_sm1(df):
    return df["GATS1i"] * df["SM1_Dz"]


PROPOSALS = [
    ("mlogp_poly", mlogp_poly),
    ("unsaturated_carbons", unsaturated_carbons),
    ("has_unsaturation", has_unsaturation),
    ("mlogp_x_sm1", mlogp_x_sm1),
    ("mlogp_per_cic", mlogp_per_cic),
    ("gats_x_sm1", gats_x_sm1),
]
