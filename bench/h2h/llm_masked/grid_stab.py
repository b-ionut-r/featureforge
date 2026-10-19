"""Domain features for t04, written from the schema and description only."""
import numpy as np
import pandas as pd

# OUTSIDE KNOWLEDGE: the ranges match the simulated 4-node star "electrical grid stability" (decentral smart grid
# control) data: f_1..f_4 reaction times tau (0.5-10 s), f_5 producer power p1 = -(p2+p3+p4), f_6..f_8 consumer
# powers p2..p4 (negative), f_9..f_12 price elasticity coefficients gamma (0.05-1). Target is likely the stability
# margin (max real part of the characteristic equation root).

_TAU = ["f_1", "f_2", "f_3", "f_4"]
_P = ["f_5", "f_6", "f_7", "f_8"]
_G = ["f_9", "f_10", "f_11", "f_12"]


def _n(df, cols):
    return df[cols].apply(pd.to_numeric, errors="coerce")


# Per-node delay times elasticity: the delayed price feedback strength that destabilises each node.
def tau_gamma_products(df):
    t = _n(df, _TAU).values
    g = _n(df, _G).values
    tg = t * g
    out = pd.DataFrame(tg, index=df.index, columns=[f"tau_g_{i + 1}" for i in range(4)])
    out["tau_g_sum"] = tg.sum(axis=1)
    out["tau_g_max"] = tg.max(axis=1)
    out["tau_g_consumers"] = tg[:, 1:].sum(axis=1)
    return out


# Power-weighted feedback: |p| * gamma * tau per node, the effective loop gain magnitude.
def power_weighted_gain(df):
    t = _n(df, _TAU).values
    g = _n(df, _G).values
    p = np.abs(_n(df, _P).values)
    pg = p * g
    ptg = p * t * g
    out = pd.DataFrame(ptg, index=df.index, columns=[f"p_tau_g_{i + 1}" for i in range(4)])
    out["p_g_sum"] = pg.sum(axis=1)
    out["p_tau_g_sum"] = ptg.sum(axis=1)
    out["p_tau_g_max"] = ptg.max(axis=1)
    return out


# Aggregate summaries of the three groups (delays, powers, elasticities).
def group_summaries(df):
    t = _n(df, _TAU)
    g = _n(df, _G)
    p = _n(df, _P[1:])
    return pd.DataFrame({
        "tau_mean": t.mean(axis=1), "tau_max": t.max(axis=1), "tau_std": t.std(axis=1),
        "g_mean": g.mean(axis=1), "g_max": g.max(axis=1), "g_std": g.std(axis=1),
        "cons_p_std": p.std(axis=1), "cons_p_min": p.min(axis=1),
        "power_balance_residual": _n(df, _P).sum(axis=1),
    })


# Producer vs consumers: the central node interacts with all; contrast its delay/elasticity with the consumers'.
def producer_vs_consumers(df):
    t = _n(df, _TAU)
    g = _n(df, _G)
    return pd.DataFrame({
        "tau1_minus_cons": t["f_1"] - t[_TAU[1:]].mean(axis=1),
        "g1_minus_cons": g["f_9"] - g[_G[1:]].mean(axis=1),
        "tau1_over_cons": t["f_1"] / t[_TAU[1:]].mean(axis=1).replace(0, np.nan),
        "g1_over_cons": g["f_9"] / g[_G[1:]].mean(axis=1).replace(0, np.nan),
    })


# Elasticity over delay: fast, strongly reacting nodes vs slow ones (phase lag of the price signal).
def g_over_tau(df):
    t = _n(df, _TAU).values
    g = _n(df, _G).values
    r = g / np.where(t == 0, np.nan, t)
    out = pd.DataFrame(r, index=df.index, columns=[f"g_over_tau_{i + 1}" for i in range(4)])
    out["g_over_tau_sum"] = np.nansum(r, axis=1)
    return out


PROPOSALS = [
    ("tau_gamma_products", tau_gamma_products),
    ("power_weighted_gain", power_weighted_gain),
    ("group_summaries", group_summaries),
    ("producer_vs_consumers", producer_vs_consumers),
    ("g_over_tau", g_over_tau),
]
