"""Domain features for t09, written from the schema and description only.

Columns are anonymous. From the values: f_1 looks like a group/entity id (repeating integers), f_2 <= f_3 look like
a start and end position (or a lower and upper cumulative amount), f_4 is a flag and f_5 a count/size. Features
are generic: interval length and ratios, group-level summaries over f_1, and simple transforms.
"""
import numpy as np
import pandas as pd


def _n(df, c):
    return pd.to_numeric(df[c], errors="coerce")


# Interval view of f_2/f_3: span, relative position and the zero-start case.
def interval(df):
    a = _n(df, "f_2")
    b = _n(df, "f_3")
    span = b - a
    return pd.DataFrame({
        "span": span,
        "log_span": np.log1p(span.clip(lower=0)),
        "start_over_end": a / b.replace(0, np.nan),
        "start_is_zero": (a == 0).astype(float),
        "log_end": np.log1p(b.clip(lower=0)),
    })


# Size relative to the interval: density of f_5 per unit of span and per unit of end value.
def density(df):
    a = _n(df, "f_2")
    b = _n(df, "f_3")
    c = _n(df, "f_5")
    span = (b - a)
    return pd.DataFrame({
        "f5_per_span": c / span.replace(0, np.nan),
        "log_f5": np.log1p(c.clip(lower=0)),
        "log_f5_minus_log_span": np.log1p(c.clip(lower=0)) - np.log1p(span.clip(lower=0)),
    })


# Group summaries over the id-like column f_1 (label-free): group size, position within group, group extents.
def group_stats(df):
    g = _n(df, "f_1")
    a = _n(df, "f_2")
    b = _n(df, "f_3")
    c = _n(df, "f_5")
    flag = _n(df, "f_4")
    tmp = pd.DataFrame({"g": g, "a": a, "b": b, "c": c, "flag": flag})
    grp = tmp.groupby("g")
    size = grp["a"].transform("size").astype(float)
    return pd.DataFrame({
        "grp_size": size,
        "grp_rank_start": grp["a"].rank(method="average") / size,
        "grp_max_end": grp["b"].transform("max"),
        "end_over_grp_max": b / grp["b"].transform("max").replace(0, np.nan),
        "grp_f5_mean": grp["c"].transform("mean"),
        "f5_over_grp_mean": c / grp["c"].transform("mean").replace(0, np.nan),
        "grp_flag_mean": grp["flag"].transform("mean"),
    })


# Flag interactions: whether the binary flag changes the meaning of size and span.
def flag_interactions(df):
    f = _n(df, "f_4")
    c = _n(df, "f_5")
    span = _n(df, "f_3") - _n(df, "f_2")
    return pd.DataFrame({
        "flag_x_logf5": f * np.log1p(c.clip(lower=0)),
        "flag_x_logspan": f * np.log1p(span.clip(lower=0)),
    })


# Neighbour gap: within each f_1 group, gap from this row's start to the previous row's end (sorted by start).
def neighbour_gap(df):
    tmp = pd.DataFrame({"g": _n(df, "f_1"), "a": _n(df, "f_2"), "b": _n(df, "f_3")})
    tmp["_i"] = np.arange(len(tmp))
    s = tmp.sort_values(["g", "a", "b"])
    prev_end = s.groupby("g")["b"].shift(1)
    next_start = s.groupby("g")["a"].shift(-1)
    s["gap_prev"] = s["a"] - prev_end
    s["gap_next"] = next_start - s["b"]
    s = s.sort_values("_i")
    return pd.DataFrame({"gap_prev": s["gap_prev"].values, "gap_next": s["gap_next"].values}, index=df.index)


PROPOSALS = [
    ("interval", interval),
    ("density", density),
    ("group_stats", group_stats),
    ("flag_interactions", flag_interactions),
    ("neighbour_gap", neighbour_gap),
]
