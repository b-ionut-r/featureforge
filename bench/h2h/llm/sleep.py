"""Domain features for sleep, written from the schema and description only."""
import numpy as np
import pandas as pd

V = [f"V{i}" for i in range(13)]


def row_mean(df):
    # V0..V12 look like 0-10 binned signal levels (e.g. EEG band powers); overall level indicates arousal
    return df[V].mean(axis=1)


def row_std(df):
    # spread across channels/bands: flat profile vs peaked spectrum
    return df[V].std(axis=1)


def zero_count(df):
    # number of channels at their floor value
    return (df[V] == 0).sum(axis=1).astype(float)


def low_high_contrast(df):
    # if columns are ordered bands, contrast of low-index vs high-index ones mimics slow-wave vs fast activity
    return df[V[1:6]].mean(axis=1) - df[V[7:13]].mean(axis=1)


def argmax_channel(df):
    # dominant channel/band
    return pd.Series(df[V].values.argmax(axis=1).astype(float), index=df.index)


PROPOSALS = [("row_mean", row_mean), ("low_high_contrast", low_high_contrast), ("row_std", row_std),
             ("zero_count", zero_count), ("argmax_channel", argmax_channel)]
