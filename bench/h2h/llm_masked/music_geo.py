"""Domain features for data, written from the schema and description only.

Columns are anonymous, so these are generic features. The schema shows a repeating structure: four blocks of
29 columns (f_1-f_29, f_30-f_58, f_59-f_87, f_88-f_116); in each block the first 17 columns are distinct and the
last 12 are exact copies of one value. Features summarise blocks and compare corresponding positions across blocks.
"""
import numpy as np
import pandas as pd

_STARTS = [1, 30, 59, 88]


def _block(df, b):
    s = _STARTS[b]
    return df[["f_%d" % i for i in range(s, s + 17)]]


def _shared(df, b):
    return df["f_%d" % (_STARTS[b] + 17)]


def block_means(df):
    # level of each block's 17 distinct measurements
    return pd.DataFrame({"blk%d_mean" % b: _block(df, b).mean(axis=1) for b in range(4)})


def block_spread(df):
    # within-block dispersion and extremes
    out = {}
    for b in range(4):
        x = _block(df, b)
        out["blk%d_std" % b] = x.std(axis=1)
        out["blk%d_max" % b] = x.max(axis=1)
        out["blk%d_min" % b] = x.min(axis=1)
    return pd.DataFrame(out)


def block_trend(df):
    # change between consecutive blocks (if blocks are repeated measurements, e.g. over time)
    m = block_means(df).values
    out = {"blk_d%d" % i: m[:, i + 1] - m[:, i] for i in range(3)}
    out["blk_last_minus_first"] = m[:, 3] - m[:, 0]
    return pd.DataFrame(out, index=df.index)


def position_across_blocks(df):
    # mean over the 4 blocks of each of the 17 positions: denoised per-position value
    arr = np.stack([_block(df, b).values for b in range(4)], axis=0)
    return pd.DataFrame(arr.mean(axis=0), index=df.index, columns=["pos%d_mean" % j for j in range(17)])


def shared_columns(df):
    # the repeated column of each block: mean and trend
    sh = pd.DataFrame({"sh%d" % b: _shared(df, b) for b in range(4)})
    return pd.DataFrame({
        "shared_mean": sh.mean(axis=1),
        "shared_trend": sh["sh3"] - sh["sh0"],
        "shared_std": sh.std(axis=1),
    })


def global_summary(df):
    # overall row summary across all distinct columns
    x = pd.concat([_block(df, b) for b in range(4)], axis=1)
    return pd.DataFrame({
        "all_mean": x.mean(axis=1),
        "all_std": x.std(axis=1),
        "all_absmean": x.abs().mean(axis=1),
        "n_pos": (x > 0).sum(axis=1).astype(float),
    })


PROPOSALS = [
    ("block_means", block_means),
    ("block_spread", block_spread),
    ("block_trend", block_trend),
    ("position_across_blocks", position_across_blocks),
    ("shared_columns", shared_columns),
    ("global_summary", global_summary),
]
