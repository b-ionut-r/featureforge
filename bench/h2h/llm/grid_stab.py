"""Domain features for grid_stab, written from the schema and description only."""
import numpy as np
import pandas as pd

_TAU = ["tau1", "tau2", "tau3", "tau4"]
_G = ["g1", "g2", "g3", "g4"]
_P = ["p2", "p3", "p4"]


# Slow reactions destabilise: summary of reaction delays.
def tau_summary(df):
    t = df[_TAU].astype(float)
    return pd.DataFrame({
        "tau_mean": t.mean(axis=1),
        "tau_max": t.max(axis=1),
        "tau_cons_mean": t[["tau2", "tau3", "tau4"]].mean(axis=1),
    }, index=df.index)


# Price elasticity summary: stronger price feedback amplifies delayed responses.
def g_summary(df):
    g = df[_G].astype(float)
    return pd.DataFrame({"g_mean": g.mean(axis=1), "g_max": g.max(axis=1)}, index=df.index)


# Delayed feedback strength: elasticity times reaction delay per participant (gain x delay drives oscillation).
def gain_delay(df):
    out = {}
    for i in range(1, 5):
        out["gt%d" % i] = df["g%d" % i] * df["tau%d" % i]
    out = pd.DataFrame(out, index=df.index)
    out["gt_sum"] = out.sum(axis=1)
    out["gt_max"] = out[["gt1", "gt2", "gt3", "gt4"]].max(axis=1)
    return out


# Consumer-power-weighted delay and elasticity: large consumers with slow, elastic response matter more.
def power_weighted(df):
    w = df[_P].abs().astype(float)
    ws = w.sum(axis=1).replace(0, np.nan)
    tau_w = (w.values * df[["tau2", "tau3", "tau4"]].values).sum(axis=1) / ws
    g_w = (w.values * df[["g2", "g3", "g4"]].values).sum(axis=1) / ws
    gt_w = (w.values * (df[["g2", "g3", "g4"]].values * df[["tau2", "tau3", "tau4"]].values)).sum(axis=1) / ws
    return pd.DataFrame({"tau_pw": tau_w, "g_pw": g_w, "gt_pw": gt_w}, index=df.index)


# Imbalance across consumers (heterogeneous loads).
def consumer_imbalance(df):
    p = df[_P].astype(float)
    return pd.DataFrame({"p_std": p.std(axis=1), "p_min": p.min(axis=1)}, index=df.index)


# Heterogeneity of delays among participants (desynchronised responses).
def tau_spread(df):
    t = df[_TAU].astype(float)
    return t.std(axis=1)


# Producer relative to consumers: generator delay and elasticity versus consumer average.
def producer_vs_consumers(df):
    return pd.DataFrame({
        "tau1_rel": df["tau1"] - df[["tau2", "tau3", "tau4"]].mean(axis=1),
        "g1_rel": df["g1"] - df[["g2", "g3", "g4"]].mean(axis=1),
        "p1_g1": df["p1"] * df["g1"],
    }, index=df.index)


PROPOSALS = [
    ("tau_summary", tau_summary),
    ("g_summary", g_summary),
    ("gain_delay", gain_delay),
    ("power_weighted", power_weighted),
    ("consumer_imbalance", consumer_imbalance),
    ("tau_spread", tau_spread),
    ("producer_vs_consumers", producer_vs_consumers),
]
