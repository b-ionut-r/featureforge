"""Domain features for wall_robot, written from the schema and description only."""
import numpy as np
import pandas as pd

# Sensor Vi sits at angle 180 - 15*(i-1) degrees, wrapped to (-180, 180]: V1 front (180), V7 at -90,
# V13 back (0), V19 at +90. 5.0 appears to be the max-range (no echo) reading.
COLS = ["V%d" % i for i in range(1, 25)]


def _s(df, idx):
    return df[["V%d" % i for i in idx]]


# Front sector (V1 +/- 30 deg): distance to an obstacle ahead decides whether to turn sharply.
def front_sector(df):
    f = _s(df, [23, 24, 1, 2, 3])
    return pd.DataFrame({"front_min": f.min(axis=1), "front_mean": f.mean(axis=1), "front_V1": df["V1"]})


# Side sectors around -90 deg (V5..V9) and +90 deg (V17..V21): distance to the followed wall on each side.
def side_sectors(df):
    a = _s(df, [5, 6, 7, 8, 9])
    b = _s(df, [17, 18, 19, 20, 21])
    return pd.DataFrame({
        "sideA_min": a.min(axis=1), "sideB_min": b.min(axis=1),
        "side_min_diff": a.min(axis=1) - b.min(axis=1),
        "side_mean_ratio": a.mean(axis=1) / b.mean(axis=1).replace(0, np.nan),
    })


# Front-diagonal sectors (+/-45..60 deg off the front): an approaching corner or opening shows up here first.
def diagonal_sectors(df):
    da = _s(df, [3, 4, 5])
    db = _s(df, [21, 22, 23])
    return pd.DataFrame({
        "diagA_min": da.min(axis=1), "diagB_min": db.min(axis=1),
        "diag_diff": da.min(axis=1) - db.min(axis=1),
    })


# Wall angle relative to heading: difference between front-side and rear-side readings on each flank.
def wall_alignment(df):
    return pd.DataFrame({
        "alignA": df["V6"] - df["V8"],
        "alignB": df["V20"] - df["V18"],
        "alignA2": _s(df, [4, 5, 6]).mean(axis=1) - _s(df, [8, 9, 10]).mean(axis=1),
        "alignB2": _s(df, [20, 21, 22]).mean(axis=1) - _s(df, [16, 17, 18]).mean(axis=1),
    })


# Global ring summaries: closest obstacle, its bearing (as sin/cos), and number of no-echo sensors.
def ring_summary(df):
    x = df[COLS].values.astype(float)
    ang = np.deg2rad(180 - 15 * np.arange(24))
    am = np.nanargmin(np.where(np.isnan(x), np.inf, x), axis=1)
    return pd.DataFrame({
        "ring_min": np.nanmin(x, axis=1),
        "ring_mean": np.nanmean(x, axis=1),
        "ring_std": np.nanstd(x, axis=1),
        "closest_sin": np.sin(ang[am]),
        "closest_cos": np.cos(ang[am]),
        "n_maxrange": (x >= 4.99).sum(axis=1).astype(float),
    }, index=df.index)


# Inverse-distance weighted repulsion vector (potential-field style): direction the obstacles push the robot.
def repulsion_vector(df):
    x = df[COLS].values.astype(float)
    ang = np.deg2rad(180 - 15 * np.arange(24))
    w = 1.0 / np.clip(x, 0.05, None)
    return pd.DataFrame({
        "rep_x": (w * np.cos(ang)).sum(axis=1),
        "rep_y": (w * np.sin(ang)).sum(axis=1),
    }, index=df.index)


# Front clearance relative to side clearance: blocked front with a close wall means a sharp turn is needed.
def front_vs_side(df):
    f = _s(df, [24, 1, 2]).min(axis=1)
    a = _s(df, [6, 7, 8]).min(axis=1)
    b = _s(df, [18, 19, 20]).min(axis=1)
    return pd.DataFrame({
        "front_over_sideA": f / a.replace(0, np.nan),
        "front_over_sideB": f / b.replace(0, np.nan),
    })


PROPOSALS = [
    ("front_sector", front_sector),
    ("side_sectors", side_sectors),
    ("diagonal_sectors", diagonal_sectors),
    ("wall_alignment", wall_alignment),
    ("ring_summary", ring_summary),
    ("repulsion_vector", repulsion_vector),
    ("front_vs_side", front_vs_side),
]
