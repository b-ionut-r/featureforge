"""Domain features for mv, written from the schema and description only.

The generating rules are given in the description; features replicate the branch conditions and branch formulas.
"""
import numpy as np
import pandas as pd


def _flags(df):
    return (df["x7"].astype(str) == "yes"), (df["x8"].astype(str) == "normal")


def branch_id(df):
    # which rule of the generator applies (X2>2 is impossible given X2 in [-15,-10], so 4 live branches)
    yes, normal = _flags(df)
    mid = df["x4"].between(-2, 2)
    return pd.Series(np.select([df["x2"] > 2, mid, yes, normal], [0, 1, 2, 3], 4), index=df.index).astype(float)


def branch_formulas(df):
    # each branch's formula value: 10-2X1, 3-X1/X4, X6+X1, X1/2
    return pd.DataFrame({"f_mid": 10 - 2 * df["x1"], "f_yes": 3 - df["x1"] / df["x4"].replace(0, np.nan),
                         "f_normal": df["x6"] + df["x1"], "f_large": df["x1"] / 2, "f_x2": 35 - 0.5 * df["x4"]})


def rule_prediction(df):
    # the full piecewise rule as one feature
    yes, normal = _flags(df)
    f = branch_formulas(df)
    return pd.Series(np.select([df["x2"] > 2, df["x4"].between(-2, 2), yes, normal],
                               [f["f_x2"], f["f_mid"], f["f_yes"], f["f_normal"]], f["f_large"]), index=df.index)


def flags(df):
    # categorical flags as numbers; |X4|<=2 indicator
    yes, normal = _flags(df)
    return pd.DataFrame({"x7_yes": yes.astype(float), "x8_normal": normal.astype(float),
                         "x4_mid": df["x4"].between(-2, 2).astype(float),
                         "x3_green": (df["x3"].astype(str) == "green").astype(float),
                         "x3_red": (df["x3"].astype(str) == "red").astype(float)})


def eps(df):
    # X6 = X4 * eps: recover the noise multiplier
    return df["x6"] / df["x4"].replace(0, np.nan)


PROPOSALS = [("branch_id", branch_id), ("branch_formulas", branch_formulas), ("rule_prediction", rule_prediction),
             ("flags", flags), ("eps", eps)]
