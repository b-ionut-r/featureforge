"""Domain features for wine_quality, written from the schema and description only."""
import numpy as np
import pandas as pd


def free_so2_ratio(df):
    # share of SO2 that is free (protective) vs bound; low share means oxidised/heavily treated wine
    return df["free.sulfur.dioxide"] / df["total.sulfur.dioxide"].replace(0, np.nan)


# OUTSIDE KNOWLEDGE: molecular (active) SO2 = free SO2 / (1 + 10^(pH - 1.81)), standard enology formula.
def molecular_so2(df):
    # antimicrobial molecular SO2 depends on pH; too much smells, too little spoils
    return df["free.sulfur.dioxide"] / (1 + 10 ** (df["pH"] - 1.81))


def total_acidity(df):
    # total acidity and the volatile fraction (acetic acid, a vinegar fault)
    tot = df["fixed.acidity"] + df["volatile.acidity"] + df["citric.acid"]
    return pd.DataFrame({"total_acid": tot, "volatile_share": df["volatile.acidity"] / tot.replace(0, np.nan)})


def sugar_alcohol(df):
    # sweetness/alcohol balance: residual sugar vs alcohol from fermented sugar
    return pd.DataFrame({"sugar_per_alc": df["residual.sugar"] / df["alcohol"].replace(0, np.nan),
                         "log_sugar": np.log1p(df["residual.sugar"].clip(lower=0))})


# OUTSIDE KNOWLEDGE: density of wine ~ 1 + k*sugar - k'*alcohol; residual from that captures extract/body.
def density_residual(df):
    # density not explained by sugar and alcohol = dry extract / body
    return df["density"] - (0.9965 + 0.00037 * df["residual.sugar"] - 0.0012 * (df["alcohol"] - 10))


def alcohol_x_volatile(df):
    # alcohol is the main quality driver; volatile acidity the main fault: their contrast
    return pd.DataFrame({"alc_minus_va": df["alcohol"] - 10 * df["volatile.acidity"],
                         "alc_x_sulph": df["alcohol"] * df["sulphates"]})


def red_white_proxy(df):
    # red vs white: reds have higher chlorides/volatile acidity, whites much more total SO2
    return np.log(df["total.sulfur.dioxide"].clip(lower=1)) - 20 * df["chlorides"] - 2 * df["volatile.acidity"]


def acid_ph(df):
    # buffering: pH relative to fixed acidity; sulphates per chloride (salty vs preserved)
    return pd.DataFrame({"ph_x_acid": df["pH"] * df["fixed.acidity"],
                         "sulph_per_chlor": df["sulphates"] / df["chlorides"].replace(0, np.nan)})


PROPOSALS = [("free_so2_ratio", free_so2_ratio), ("molecular_so2", molecular_so2), ("total_acidity", total_acidity),
             ("sugar_alcohol", sugar_alcohol), ("density_residual", density_residual),
             ("alcohol_x_volatile", alcohol_x_volatile), ("red_white_proxy", red_white_proxy), ("acid_ph", acid_ph)]
