"""Domain features for planes2d, written from the schema and description only.

The description states the generating rule (two linear planes selected by x1); features encode it directly.
"""
import numpy as np
import pandas as pd


def _f(df, c):
    return pd.to_numeric(df[c], errors="coerce").astype(float)


def plane_pos(df):
    # description: when x1 = 1 the response is 3 + 3*x2 + 2*x3 + x4
    return 3 + 3 * _f(df, "x2") + 2 * _f(df, "x3") + _f(df, "x4")


def plane_neg(df):
    # description: when x1 = -1 the response is -3 + 3*x5 + 2*x6 + x7
    return -3 + 3 * _f(df, "x5") + 2 * _f(df, "x6") + _f(df, "x7")


def planes_rule(df):
    # description: the full piecewise rule, choosing the plane by the sign of x1
    x1 = _f(df, "x1")
    return pd.Series(np.where(x1 > 0, plane_pos(df), np.where(x1 < 0, plane_neg(df), np.nan)), index=df.index)


def gated_terms(df):
    # each branch's linear part gated by x1, so a linear model can use both planes
    x1 = _f(df, "x1")
    pos = (x1 > 0).astype(float)
    return pd.DataFrame({"gate_pos_lin": pos * (3 * _f(df, "x2") + 2 * _f(df, "x3") + _f(df, "x4")),
                         "gate_neg_lin": (1 - pos) * (3 * _f(df, "x5") + 2 * _f(df, "x6") + _f(df, "x7"))})


def x1_interactions(df):
    # x1 times each relevant variable: sign-flip interactions that select the active plane
    x1 = _f(df, "x1")
    return pd.DataFrame({f"x1_x{i}": x1 * _f(df, f"x{i}") for i in range(2, 8)})


PROPOSALS = [("planes_rule", planes_rule), ("gated_terms", gated_terms), ("plane_pos", plane_pos),
             ("plane_neg", plane_neg), ("x1_interactions", x1_interactions)]
