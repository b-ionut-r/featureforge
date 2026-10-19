"""Domain features for phoneme, written from the schema and description only."""
import numpy as np
import pandas as pd

COLS = ["Aa", "Ao", "Dcl", "Iy", "Sh"]


def pc_norm(df):
    # the 5 columns are leading cochlear-spectrum principal components; overall energy separates nasal vs oral vowels
    return np.sqrt((df[COLS].astype(float) ** 2).sum(axis=1))


def aa_ao_ratio(df):
    # balance between the two dominant components acts as a spectral-tilt / vowel-openness cue
    return df["Aa"] / (df["Aa"].abs() + df["Ao"].abs()).replace(0, np.nan)


def aa_minus_ao(df):
    # contrast of the first two components (an oblique split trees approximate poorly)
    return df["Aa"] - df["Ao"]


def aa_times_ao(df):
    # interaction of the two dominant components
    return df["Aa"] * df["Ao"]


def minor_energy_share(df):
    # share of energy outside the first two components: fine spectral detail
    sq = df[COLS].astype(float) ** 2
    return sq[["Dcl", "Iy", "Sh"]].sum(axis=1) / sq.sum(axis=1).replace(0, np.nan)


PROPOSALS = [("pc_norm", pc_norm), ("aa_ao_ratio", aa_ao_ratio), ("aa_minus_ao", aa_minus_ao),
             ("aa_times_ao", aa_times_ao), ("minor_energy_share", minor_energy_share)]
