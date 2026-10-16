"""Domain features for miniboone, written from the schema and description only."""
import numpy as np
import pandas as pd

# 50 anonymous particle-ID variables from a Cherenkov detector; electron (signal) vs muon events.


def _cols(df):
    return [c for c in df.columns if c.startswith("ParticleID_")]


def _clean(df):
    x = df[_cols(df)].apply(pd.to_numeric, errors="coerce")
    return x.where(x > -900)  # -999 is the detector's "not reconstructed" sentinel


def sentinel_count(df):
    # events where reconstruction failed (all -999) behave completely differently
    x = df[_cols(df)].apply(pd.to_numeric, errors="coerce")
    return (x <= -900).sum(axis=1).astype(float)


def rank_mean_profile(df):
    # average rank-standardised value across all 50 PID variables: a crude 1st principal component
    r = _clean(df).rank(pct=True)
    return pd.DataFrame({"rank_mean": r.mean(axis=1), "rank_std": r.std(axis=1)}, index=df.index)


def energy_normalised(df):
    # large-scale variables (likely charge/energy sums) normalised by ParticleID_15 to remove event energy
    x = _clean(df)
    d = lambda a, b: x[a] / x[b].where(x[b].abs() > 1e-9)
    return pd.DataFrame({"p2_over_p15": d("ParticleID_2", "ParticleID_15"),
                         "p22_over_p15": d("ParticleID_22", "ParticleID_15"),
                         "p17_over_p15": d("ParticleID_17", "ParticleID_15"),
                         "p26_over_p33": d("ParticleID_26", "ParticleID_33"),
                         "p40_over_p33": d("ParticleID_40", "ParticleID_33")}, index=df.index)


def log_heavy_tails(df):
    # products of shape variables: electrons give fuzzy rings, muons sharp rings (shape x size)
    x = _clean(df)
    return pd.DataFrame({"p0_x_p8": x["ParticleID_0"] * x["ParticleID_8"],
                         "p3_x_p24": x["ParticleID_3"] * x["ParticleID_24"],
                         "p0_minus_p10": x["ParticleID_0"] - x["ParticleID_10"]}, index=df.index)


PROPOSALS = [
    ("sentinel_count", sentinel_count),
    ("energy_normalised", energy_normalised),
    ("rank_mean_profile", rank_mean_profile),
    ("log_heavy_tails", log_heavy_tails),
]
