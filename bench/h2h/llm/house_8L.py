"""Domain features for house_8L, written from the schema and description only.

Column codes are only partly decodable (P* = person counts/proportions, H* = housing; P3 looks like a population
count, H15p1 like rooms per unit, others are proportions), so features are mostly generic but sensible.
"""
import numpy as np
import pandas as pd


def log_pop(df):
    # P3 is a heavy-tailed count (place size): log scale; big places have pricier housing markets
    return np.log1p(df["P3"].clip(lower=0))


def logit_props(df):
    # proportions near 0/1 are better on a logit scale
    out = {}
    for c in ["P6p4", "P11p3", "P16p2", "P19p2", "H5p2", "H40p4"]:
        p = df[c].clip(0.005, 0.995)
        out[f"logit_{c}"] = np.log(p / (1 - p))
    return pd.DataFrame(out, index=df.index)


def rooms_terms(df):
    # H15p1 resembles average rooms per unit: larger homes cost more; square for convexity
    return pd.DataFrame({"h15_sq": df["H15p1"] ** 2, "h15_x_logpop": df["H15p1"] * np.log1p(df["P3"].clip(lower=0))})


def prop_contrasts(df):
    # generic contrasts between person proportions
    return pd.DataFrame({"p16_minus_p11": df["P16p2"] - df["P11p3"],
                         "p19_over_p16": df["P19p2"] / df["P16p2"].replace(0, np.nan)})


def housing_x_people(df):
    # generic housing-by-demographic interactions
    return pd.DataFrame({"h5_x_p19": df["H5p2"] * df["P19p2"], "h40_x_p6": df["H40p4"] * df["P6p4"],
                         "h40_x_h5": df["H40p4"] * df["H5p2"]})


def small_place(df):
    # very small places give noisy proportions: flag them
    return (df["P3"] < 200).astype(float)


PROPOSALS = [("log_pop", log_pop), ("logit_props", logit_props), ("rooms_terms", rooms_terms),
             ("prop_contrasts", prop_contrasts), ("housing_x_people", housing_x_people), ("small_place", small_place)]
