"""Domain features for pol, written from the schema and description only."""
import numpy as np
import pandas as pd

BLOCK = [f"f{i}" for i in range(13, 34)]


def block_sum(df):
    # f13..f33 look like a sparse count/allocation vector; its total load is a natural summary
    return df[[c for c in BLOCK if c in df]].sum(axis=1)


def block_nonzero(df):
    # how many of the sparse slots are active
    return (df[[c for c in BLOCK if c in df]] > 0).sum(axis=1).astype(float)


def block_max_share(df):
    # concentration: largest slot relative to total load
    b = df[[c for c in BLOCK if c in df]]
    return b.max(axis=1) / b.sum(axis=1).replace(0, np.nan)


def head_mean(df):
    # mean of the varying f5..f9 levels
    return df[["f5", "f6", "f7", "f8", "f9"]].mean(axis=1)


def f5_minus_f6(df):
    # difference of the two most variable level columns
    return df["f5"] - df["f6"]


PROPOSALS = [("block_sum", block_sum), ("block_nonzero", block_nonzero), ("block_max_share", block_max_share),
             ("head_mean", head_mean), ("f5_minus_f6", f5_minus_f6)]
