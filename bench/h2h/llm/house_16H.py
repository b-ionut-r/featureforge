"""Domain features for house_16H, written from the schema and description only."""
import numpy as np
import pandas as pd

# 1990 US census block-group data (house_16H from the Delve "census-house" family): P1 is a population
# count, the other columns are proportions (P* = person attributes, H* = housing attributes).
PROPS = ["P5p1", "P6p2", "P11p4", "P14p9", "P15p1", "P15p3", "P16p2", "P18p2", "P27p4",
         "H2p2", "H8p2", "H10p1", "H13p1", "H18pA", "H40p4"]


def log_population(df):
    # area size on a log scale; dense urban block groups have different house prices
    return np.log1p(df["P1"].clip(lower=0).astype(float))


def proportion_counts(df):
    # proportions times population give absolute counts, which the model cannot form itself
    p1 = df["P1"].astype(float)
    return pd.DataFrame({f"{c}_count": np.log1p((df[c].astype(float) * p1).clip(lower=0))
                         for c in ["P6p2", "P15p3", "H2p2", "H8p2", "H18pA"]}, index=df.index)


def proportion_ratios(df):
    # pairwise ratios of the strongest-varying proportions (minority share vs owner/vacancy shares)
    eps = 1e-3
    return pd.DataFrame({
        "P6p2_over_H10p1": df["P6p2"] / (df["H10p1"] + eps),
        "P11p4_over_P14p9": df["P11p4"] / (df["P14p9"] + eps),
        "H2p2_over_H13p1": df["H2p2"] / (df["H13p1"] + eps),
        "P15p3_minus_P6p2": df["P15p3"] - df["P6p2"],
    }, index=df.index).astype(float)


def proportion_summary(df):
    # overall profile: mean and spread of the proportion columns
    p = df[[c for c in PROPS if c in df.columns]].astype(float)
    return pd.DataFrame({"prop_mean": p.mean(axis=1), "prop_std": p.std(axis=1)}, index=df.index)


PROPOSALS = [
    ("proportion_ratios", proportion_ratios),
    ("proportion_counts", proportion_counts),
    ("log_population", log_population),
    ("proportion_summary", proportion_summary),
]
