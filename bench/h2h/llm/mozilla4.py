"""Domain features for mozilla4, written from the schema and description only."""
import numpy as np
import pandas as pd


# Length of the observation interval: longer exposure windows give more chance for a defect fix.
def interval_length(df):
    return (df["end"] - df["start"]).clip(lower=0)


# Log of class size: defect proneness is reported to grow sub-linearly with module size.
def log_size(df):
    return np.log1p(df["size"].clip(lower=0))


# Size-exposure product: expected defect count scales with both size and time at risk.
def size_times_log_interval(df):
    dur = (df["end"] - df["start"]).clip(lower=0)
    return np.log1p(df["size"].clip(lower=0)) * np.log1p(dur)


# Number of recorded modifications per class (recurrent events): frequently modified classes are churn-heavy.
def n_obs_per_class(df):
    return df.groupby("id")["id"].transform("count").astype(float)


# Position of the observation within its class history (0 = class creation); late observations differ from new classes.
def obs_rank_in_class(df):
    r = df.groupby("id")["start"].rank(method="first")
    n = df.groupby("id")["id"].transform("count")
    return pd.DataFrame({"obs_rank": r, "obs_rank_frac": r / n.replace(0, np.nan)})


# Size relative to the class's own typical size: growth or shrinkage of a class across modifications.
def size_vs_class_mean(df):
    m = df.groupby("id")["size"].transform("mean")
    mx = df.groupby("id")["size"].transform("max")
    mn = df.groupby("id")["size"].transform("min")
    return pd.DataFrame({
        "size_over_class_mean": df["size"] / m.replace(0, np.nan),
        "class_size_range": mx - mn,
    })


# Share of a class's observations that are defect-fix events (the given `event` flag, not the target).
def class_event_rate(df):
    g = df.groupby("id")["event"]
    return pd.DataFrame({
        "class_event_rate": g.transform("mean"),
        "class_event_count": g.transform("sum").astype(float),
    })


# Is this the creation observation (start at 0) and how far into the study window the observation lies.
def start_flags(df):
    return pd.DataFrame({
        "is_creation": (df["start"] == 0).astype(float),
        "log_start": np.log1p(df["start"].clip(lower=0)),
        "log_end": np.log1p(df["end"].clip(lower=0)),
    })


PROPOSALS = [
    ("interval_length", interval_length),
    ("log_size", log_size),
    ("size_times_log_interval", size_times_log_interval),
    ("n_obs_per_class", n_obs_per_class),
    ("obs_rank_in_class", obs_rank_in_class),
    ("size_vs_class_mean", size_vs_class_mean),
    ("class_event_rate", class_event_rate),
    ("start_flags", start_flags),
]
