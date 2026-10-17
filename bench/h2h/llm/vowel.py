"""Domain features for vowel, written from the schema and description only.

Feature_0..9 are LPC-derived log-area ratios (vocal tract shape); individually anonymous, so features are
speaker normalisation and generic contrasts between neighbouring coefficients.
"""
import numpy as np
import pandas as pd

F = [f"Feature_{i}" for i in range(10)]


def sex_flag(df):
    # sex shifts vocal tract length and hence all formant-related coefficients
    return (df["Sex"].astype(str) == "Female").astype(float)


def speaker_normalised(df):
    # speaker-independent recognition: subtract each speaker's mean coefficient vector (vocal tract normalisation)
    X = df[F].astype(float)
    return (X - X.groupby(df["Speaker_Number"].astype(str)).transform("mean")).add_suffix("_spk")


def speaker_zscore(df):
    # also scale by each speaker's spread so speakers with wider articulation are comparable
    X = df[F].astype(float)
    g = X.groupby(df["Speaker_Number"].astype(str))
    return ((X - g.transform("mean")) / g.transform("std").replace(0, np.nan)).add_suffix("_spkz")


def sex_normalised(df):
    # coarser normalisation by sex group mean
    X = df[F].astype(float)
    return (X - X.groupby(df["Sex"].astype(str)).transform("mean")).add_suffix("_sex")


def adjacent_diffs(df):
    # differences between adjacent log-area ratios describe local tract constriction changes
    X = df[F].astype(float).values
    return pd.DataFrame(np.diff(X, axis=1), index=df.index, columns=[f"d{i}{i+1}" for i in range(9)])


def low_high_energy(df):
    # front (low-order) vs back (high-order) coefficients: rough open/close and front/back vowel axes
    X = df[F].astype(float)
    return pd.DataFrame({"low_mean": X[F[:3]].mean(axis=1), "high_mean": X[F[5:]].mean(axis=1),
                         "low_minus_high": X[F[:3]].mean(axis=1) - X[F[5:]].mean(axis=1),
                         "norm": np.sqrt((X ** 2).sum(axis=1))})


def f0_f1_ratio(df):
    # the two dominant coefficients and their ratio/angle capture the main vowel-space position
    return pd.DataFrame({"f0_x_f1": df["Feature_0"] * df["Feature_1"],
                         "angle01": np.arctan2(df["Feature_1"], df["Feature_0"])})


PROPOSALS = [("sex_flag", sex_flag), ("speaker_normalised", speaker_normalised), ("speaker_zscore", speaker_zscore),
             ("sex_normalised", sex_normalised), ("adjacent_diffs", adjacent_diffs),
             ("low_high_energy", low_high_energy), ("f0_f1_ratio", f0_f1_ratio)]
