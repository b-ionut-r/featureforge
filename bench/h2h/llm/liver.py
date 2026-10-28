"""Domain features for liver, written from the schema and description only."""
import numpy as np
import pandas as pd


# De Ritis ratio (AST/ALT): values above ~2 are characteristic of alcoholic liver injury.
# OUTSIDE KNOWLEDGE: AST/ALT > 2 is a classic marker of alcohol-related liver disease.
def de_ritis(df):
    return df["sgot"].astype(float) / df["sgpt"].astype(float).replace(0, np.nan)


# GGT is the most alcohol-sensitive enzyme and is right-skewed: use its log.
# OUTSIDE KNOWLEDGE: gamma-GT rises with heavy alcohol intake.
def log_gammagt(df):
    return np.log1p(df["gammagt"].astype(float).clip(lower=0))


# MCV above ~95 fL (macrocytosis) is associated with chronic heavy drinking.
# OUTSIDE KNOWLEDGE: macrocytosis (MCV > ~95-100 fL) is a marker of chronic alcohol use.
def mcv_excess(df):
    m = df["mcv"].astype(float)
    return pd.DataFrame({"mcv_over_95": (m - 95).clip(lower=0), "macrocytic": (m >= 95).astype(float)}, index=df.index)


# GGT relative to ALP: GGT raised without ALP points to alcohol rather than cholestasis.
def ggt_over_alp(df):
    return np.log1p(df["gammagt"].astype(float)) - np.log1p(df["alkphos"].astype(float))


# Combined enzyme burden: mean of log transaminases and GGT.
def enzyme_burden(df):
    cols = ["sgpt", "sgot", "gammagt", "alkphos"]
    return np.log1p(df[cols].astype(float).clip(lower=0)).mean(axis=1)


# GGT x MCV interaction: both elevated together are a stronger drinking signature than either alone.
def ggt_mcv_interaction(df):
    return np.log1p(df["gammagt"].astype(float)) * (df["mcv"].astype(float) - 85)


# Count of tests above typical upper reference limits.
# OUTSIDE KNOWLEDGE: approximate male reference upper limits (ALT 40, AST 40, GGT 50, ALP 120, MCV 98).
def n_abnormal(df):
    return ((df["sgpt"] > 40).astype(float) + (df["sgot"] > 40).astype(float)
            + (df["gammagt"] > 50).astype(float) + (df["alkphos"] > 120).astype(float)
            + (df["mcv"] > 98).astype(float))


PROPOSALS = [
    ("de_ritis", de_ritis),
    ("log_gammagt", log_gammagt),
    ("mcv_excess", mcv_excess),
    ("ggt_over_alp", ggt_over_alp),
    ("enzyme_burden", enzyme_burden),
    ("ggt_mcv_interaction", ggt_mcv_interaction),
    ("n_abnormal", n_abnormal),
]
