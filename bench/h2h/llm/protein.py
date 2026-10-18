"""Domain features for protein, written from the schema and description only."""
import numpy as np
import pandas as pd


# Share of the total surface that is non-polar exposed: misfolded decoys expose more hydrophobics.
def nonpolar_share(df):
    return df["F2"] / df["F1"].replace(0, np.nan)


# Mass-weighted exposed area per unit surface: average mass of exposed residues.
def mass_per_area(df):
    return df["F5"] / df["F1"].replace(0, np.nan)


# Deviation from standard exposed area, normalised by size.
def deviation_per_area(df):
    return df["F6"] / df["F1"].replace(0, np.nan)


# Euclidean distance (spread) relative to surface: compactness of the structure.
def spread_per_area(df):
    return df["F7"] / df["F1"].replace(0, np.nan)


# Spread relative to the square root of surface (size-normalised radius-like term).
def spread_per_sqrt_area(df):
    return df["F7"] / np.sqrt(df["F1"].clip(lower=0)).replace(0, np.nan)


# Secondary structure penalty per unit size.
def penalty_per_area(df):
    return df["F8"] / df["F1"].replace(0, np.nan)


# Exposed non-polar part relative to the non-polar exposed area.
def f4_per_f2(df):
    return df["F4"] / df["F2"].replace(0, np.nan)


# Overall size on log scale (larger proteins allow larger RMSD).
def log_size(df):
    return pd.DataFrame({
        "log_F1": np.log1p(df["F1"].clip(lower=0)),
        "log_F5": np.log1p(df["F5"].clip(lower=0)),
    }, index=df.index)


PROPOSALS = [
    ("nonpolar_share", nonpolar_share),
    ("mass_per_area", mass_per_area),
    ("deviation_per_area", deviation_per_area),
    ("spread_per_area", spread_per_area),
    ("spread_per_sqrt_area", spread_per_sqrt_area),
    ("penalty_per_area", penalty_per_area),
    ("f4_per_f2", f4_per_f2),
    ("log_size", log_size),
]
