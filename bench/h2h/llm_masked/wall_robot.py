"""Domain features for t06, written from the schema and description only."""
import numpy as np
import pandas as pd

# OUTSIDE KNOWLEDGE: 24 distance readings capped at 5.0 match the "wall-following robot navigation" data
# (SCITOS G5): f_1..f_24 are ultrasound sensors in a ring, 15 degrees apart, f_1 at the front, going around the body.
# The classes are movement commands (move forward, slight/sharp turns), decided from where the wall/obstacles are.

_COLS = [f"f_{i}" for i in range(1, 25)]
_ANG = np.deg2rad(np.arange(24) * 15.0)  # angle of each sensor relative to the front sensor


def _m(df):
    return df[_COLS].apply(pd.to_numeric, errors="coerce")


# Sector minima: the simplified version of this task uses front/left/right/back minimum distances.
def sector_minima(df):
    x = _m(df)
    sect = {
        "front": [24, 1, 2],
        "front_wide": [23, 24, 1, 2, 3],
        "sideA": [6, 7, 8],
        "back": [12, 13, 14],
        "sideB": [18, 19, 20],
        "diagA": [3, 4, 5],
        "diagB": [21, 22, 23],
    }
    out = {}
    for k, idx in sect.items():
        out[k + "_min"] = x[[f"f_{i}" for i in idx]].min(axis=1)
    out = pd.DataFrame(out)
    out["sideA_minus_sideB"] = out["sideA_min"] - out["sideB_min"]
    out["diagA_minus_diagB"] = out["diagA_min"] - out["diagB_min"]
    return out


# Closest obstacle: distance and direction (as sin/cos of the angle) of the nearest reading.
def nearest_obstacle(df):
    x = _m(df)
    v = x.fillna(5.0).values
    am = v.argmin(axis=1)
    ang = _ANG[am]
    return pd.DataFrame({
        "global_min": v.min(axis=1),
        "nearest_sin": np.sin(ang),
        "nearest_cos": np.cos(ang),
        "nearest_idx": am.astype(float),
    }, index=df.index)


# Saturation (no echo / out of range): count of maxed-out sensors and their angular balance.
def saturation(df):
    v = _m(df).fillna(5.0).values
    sat = (v >= 4.999).astype(float)
    return pd.DataFrame({
        "n_saturated": sat.sum(axis=1),
        "sat_sin": (sat * np.sin(_ANG)).sum(axis=1),
        "sat_cos": (sat * np.cos(_ANG)).sum(axis=1),
    }, index=df.index)


# Proximity vector: inverse-distance weighted direction of obstacles (a repulsive-field summary).
def proximity_vector(df):
    v = _m(df).fillna(5.0).clip(lower=0.05).values
    w = 1.0 / v
    sx = (w * np.cos(_ANG)).sum(axis=1)
    sy = (w * np.sin(_ANG)).sum(axis=1)
    return pd.DataFrame({
        "prox_x": sx,
        "prox_y": sy,
        "prox_mag": np.sqrt(sx ** 2 + sy ** 2),
        "prox_angle_sin": sy / np.sqrt(sx ** 2 + sy ** 2 + 1e-9),
    }, index=df.index)


# Wall geometry: local differences between neighbouring sensors (wall slope/corners) on each side.
def neighbour_diffs(df):
    v = _m(df).values
    d = np.diff(np.concatenate([v, v[:, :1]], axis=1), axis=1)
    out = pd.DataFrame(d[:, [0, 1, 4, 5, 6, 7, 16, 17, 18, 19, 22, 23]], index=df.index,
                       columns=[f"diff_{i}" for i in [1, 2, 5, 6, 7, 8, 17, 18, 19, 20, 23, 24]])
    out["abs_diff_sum"] = np.nansum(np.abs(np.clip(d, -2, 2)), axis=1)
    return out


# Wall-following specific: distance ahead relative to the side distance (approaching a corner vs open space).
def front_vs_side(df):
    x = _m(df)
    front = x[["f_24", "f_1", "f_2"]].min(axis=1)
    sa = x[["f_6", "f_7", "f_8"]].min(axis=1)
    sb = x[["f_18", "f_19", "f_20"]].min(axis=1)
    return pd.DataFrame({
        "front_over_sideA": front / sa.replace(0, np.nan),
        "front_over_sideB": front / sb.replace(0, np.nan),
        "front_minus_minside": front - np.minimum(sa, sb),
    })


PROPOSALS = [
    ("sector_minima", sector_minima),
    ("nearest_obstacle", nearest_obstacle),
    ("saturation", saturation),
    ("proximity_vector", proximity_vector),
    ("neighbour_diffs", neighbour_diffs),
    ("front_vs_side", front_vs_side),
]
