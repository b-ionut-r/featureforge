"""Domain features for page_blocks, written from the schema and description only."""
import numpy as np
import pandas as pd

# Document page segments: text / horizontal line / picture / vertical line / graphic.


def _div(a, b):
    return a / b.replace(0, np.nan)


def transitions_per_length(df):
    # text has many white-black transitions per unit length (characters); lines and pictures have few
    return pd.DataFrame({"trans_per_len": _div(df["wb_trans"], df["lenght"]),
                         "trans_per_height": _div(df["wb_trans"], df["height"]),
                         "trans_per_area": _div(df["wb_trans"], df["area"])}, index=df.index)


def fill_ratio(df):
    # black pixels after run-length smoothing vs before: how much smoothing closes gaps (text vs solid)
    return pd.DataFrame({"black_over_and": _div(df["blackpix"], df["blackand"]),
                         "and_minus_black": _div(df["blackand"] - df["blackpix"], df["area"])}, index=df.index)


def log_shape(df):
    # log height/length/area: lines are extremely thin or extremely long, pictures are big
    return pd.DataFrame({"log_height": np.log1p(df["height"]), "log_len": np.log1p(df["lenght"]),
                         "log_area": np.log1p(df["area"]),
                         "log_eccen": np.log(df["eccen"].clip(lower=1e-3))}, index=df.index)


def height_vs_typical(df):
    # text lines share a common font height; distance from the most common heights flags non-text
    h = df["height"]
    cnt = h.groupby(h).transform("size").astype(float)
    return pd.DataFrame({"height_freq": cnt, "height_minus_median": h - h.median()}, index=df.index)


def run_length_stats(df):
    # mean black run length relative to block height: strokes of letters are short relative to height
    return pd.DataFrame({"meantr_over_height": _div(df["mean_tr"], df["height"]),
                         "blackpix_per_len": _div(df["blackpix"], df["lenght"])}, index=df.index)


PROPOSALS = [
    ("transitions_per_length", transitions_per_length),
    ("fill_ratio", fill_ratio),
    ("run_length_stats", run_length_stats),
    ("log_shape", log_shape),
    ("height_vs_typical", height_vs_typical),
]
