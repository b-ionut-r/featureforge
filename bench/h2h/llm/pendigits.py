"""Domain features for pendigits, written from the schema and description only."""
import numpy as np
import pandas as pd

# OUTSIDE KNOWLEDGE: the 16 inputs are 8 spatially resampled pen points stored as interleaved (x, y) pairs:
# input1=x1, input2=y1, ..., input15=x8, input16=y8 (standard UCI pendigits layout).


def _xy(df):
    X = df[[f"input{i}" for i in range(1, 17)]].apply(pd.to_numeric, errors="coerce").astype(float).values
    return X[:, 0::2], X[:, 1::2]


def path_length(df):
    # segment lengths and total trajectory length vs start-end chord: straight 1 vs curly 8
    x, y = _xy(df)
    seg = np.hypot(np.diff(x, axis=1), np.diff(y, axis=1))
    total = seg.sum(axis=1)
    chord = np.hypot(x[:, -1] - x[:, 0], y[:, -1] - y[:, 0])
    return pd.DataFrame({"path_len": total, "chord": chord, "straightness": chord / np.where(total == 0, np.nan, total),
                         "seg_std": seg.std(axis=1)}, index=df.index)


def start_end(df):
    # start-to-end displacement: closed shapes (0, 8) end near where they start
    x, y = _xy(df)
    return pd.DataFrame({"dx_end": x[:, -1] - x[:, 0], "dy_end": y[:, -1] - y[:, 0]}, index=df.index)


def directions(df):
    # direction of each pen segment as cos/sin: encodes the writing order and stroke shape
    x, y = _xy(df)
    dx, dy = np.diff(x, axis=1), np.diff(y, axis=1)
    n = np.hypot(dx, dy)
    n = np.where(n == 0, np.nan, n)
    out = {f"cos{k}": dx[:, k] / n[:, k] for k in range(7)}
    out.update({f"sin{k}": dy[:, k] / n[:, k] for k in range(7)})
    return pd.DataFrame(out, index=df.index)


def turning(df):
    # signed turning angles between consecutive segments: total curvature and winding (loops vs zigzags)
    x, y = _xy(df)
    ang = np.arctan2(np.diff(y, axis=1), np.diff(x, axis=1))
    turn = np.diff(ang, axis=1)
    turn = (turn + np.pi) % (2 * np.pi) - np.pi
    return pd.DataFrame({"turn_sum": turn.sum(axis=1), "turn_abs": np.abs(turn).sum(axis=1),
                         "turn_max": np.abs(turn).max(axis=1)}, index=df.index)


def signed_area(df):
    # shoelace area of the closed polygon: sign gives clockwise vs counter-clockwise writing, size gives loopiness
    x, y = _xy(df)
    xs, ys = np.roll(x, -1, axis=1), np.roll(y, -1, axis=1)
    return pd.Series(0.5 * (x * ys - xs * y).sum(axis=1), index=df.index)


def centroid_spread(df):
    # centroid and spread of points: where the digit's mass sits after normalisation
    x, y = _xy(df)
    return pd.DataFrame({"mx": x.mean(axis=1), "my": y.mean(axis=1), "sx": x.std(axis=1), "sy": y.std(axis=1),
                         "w_over_h": (x.max(axis=1) - x.min(axis=1)) /
                         np.where((y.max(axis=1) - y.min(axis=1)) == 0, np.nan, y.max(axis=1) - y.min(axis=1))},
                        index=df.index)


def self_proximity(df):
    # minimum distance between non-adjacent points: small when the stroke crosses or revisits itself (4, 8, 0)
    x, y = _xy(df)
    d = np.hypot(x[:, :, None] - x[:, None, :], y[:, :, None] - y[:, None, :])
    mask = np.abs(np.arange(8)[:, None] - np.arange(8)[None, :]) >= 2
    d = np.where(mask[None], d, np.inf)
    return pd.Series(d.min(axis=(1, 2)), index=df.index)


PROPOSALS = [("path_length", path_length), ("start_end", start_end), ("directions", directions), ("turning", turning),
             ("signed_area", signed_area), ("centroid_spread", centroid_spread), ("self_proximity", self_proximity)]
