"""Domain features for wine_white, written from the schema and description only."""
import numpy as np
import pandas as pd


def molecular_so2(df):
    # molecular (antimicrobial, sensorially harsh) SO2 = free / (1 + 10^(pH - 1.81))
    return df["free sulfur dioxide"] / (1 + 10 ** (df["pH"] - 1.81))


def free_so2_share(df):
    # free/total SO2: high bound SO2 signals oxidation/aldehydes, a known quality defect
    return pd.DataFrame({"free_share": df["free sulfur dioxide"] / df["total sulfur dioxide"].replace(0, np.nan),
                         "bound_so2": df["total sulfur dioxide"] - df["free sulfur dioxide"]})


def density_residual(df):
    # density not explained by alcohol and sugar = dry extract/body; fit a deterministic linear model on the inputs
    X = np.column_stack([np.ones(len(df)), df["alcohol"].values, df["residual sugar"].values])
    y = df["density"].values
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    return pd.Series((y - X @ beta) * 1000, index=df.index)


def alcohol_density(df):
    # alcohol relative to density: the classic strongest quality signal, sharpened
    return df["alcohol"] / df["density"]


def sugar_alcohol(df):
    # sweetness balance: residual sugar per unit alcohol (stuck vs complete fermentation)
    return df["residual sugar"] / df["alcohol"].replace(0, np.nan)


def total_acidity(df):
    # total acidity and the volatile share (vinegar taint)
    tot = df["fixed acidity"] + df["volatile acidity"] + df["citric acid"]
    return pd.DataFrame({"total_acid": tot, "volatile_share": df["volatile acidity"] / tot.replace(0, np.nan)})


def chloride_ratio(df):
    # saltiness relative to sulphates (both minerals, different sensory effect)
    return df["chlorides"] / df["sulphates"].replace(0, np.nan)


def acid_ph(df):
    # buffering: acidity per pH unit
    return df["fixed acidity"] / df["pH"].replace(0, np.nan)


PROPOSALS = [("alcohol_density", alcohol_density), ("density_residual", density_residual), ("molecular_so2", molecular_so2),
             ("free_so2_share", free_so2_share), ("total_acidity", total_acidity), ("sugar_alcohol", sugar_alcohol),
             ("chloride_ratio", chloride_ratio), ("acid_ph", acid_ph)]
