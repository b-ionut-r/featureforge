"""Domain features for data, written from the schema and description only."""
import numpy as np
import pandas as pd

# OUTSIDE KNOWLEDGE: the schema (345 rows; values ~85-98, then three enzyme-like columns and a skewed one)
# looks like the BUPA liver-disorders layout: f_1=MCV, f_2=alkaline phosphatase, f_3=ALT(SGPT),
# f_4=AST(SGOT), f_5=gamma-GT; the target would then be alcohol consumption.


def de_ritis(df):
    # AST/ALT ratio (De Ritis); >1-2 is classically associated with alcoholic liver injury
    # OUTSIDE KNOWLEDGE: De Ritis ratio and its alcohol association
    return (df["f_4"] / df["f_3"].replace(0, np.nan)).rename("ast_alt")


def log_ggt(df):
    # gamma-GT is right-skewed and the most alcohol-sensitive marker; log scale is more linear
    return np.log1p(df["f_5"].clip(lower=0)).rename("log_ggt")


def ggt_ratios(df):
    # GGT relative to transaminases / ALP separates alcohol-induced enzyme induction from other liver damage
    ggt = df["f_5"]
    return pd.DataFrame({
        "ggt_alt": ggt / df["f_3"].replace(0, np.nan),
        "ggt_ast": ggt / df["f_4"].replace(0, np.nan),
        "ggt_alp": ggt / df["f_2"].replace(0, np.nan),
    })


def mcv_macrocytosis(df):
    # MCV above ~95-100 fl (macrocytosis) is a chronic-drinking marker; distance from normal mean
    # OUTSIDE KNOWLEDGE: normal MCV ~80-96 fl
    return pd.DataFrame({
        "mcv_excess": (df["f_1"] - 92.0).clip(lower=0),
        "mcv_high": (df["f_1"] >= 96).astype(float),
    })


def enzyme_load(df):
    # overall liver enzyme burden: sum of logs of transaminases and GGT
    s = np.log1p(df["f_3"].clip(lower=0)) + np.log1p(df["f_4"].clip(lower=0)) + np.log1p(df["f_5"].clip(lower=0))
    return s.rename("log_enzyme_sum")


def combined_alcohol_index(df):
    # combine the two classic alcohol markers (GGT and MCV) into a single score
    z_ggt = np.log1p(df["f_5"].clip(lower=0))
    z_ggt = (z_ggt - z_ggt.mean()) / (z_ggt.std() + 1e-9)
    z_mcv = (df["f_1"] - df["f_1"].mean()) / (df["f_1"].std() + 1e-9)
    return (z_ggt + z_mcv).rename("ggt_mcv_index")


PROPOSALS = [
    ("ast_alt_ratio", de_ritis),
    ("log_ggt", log_ggt),
    ("ggt_ratios", ggt_ratios),
    ("mcv_macrocytosis", mcv_macrocytosis),
    ("enzyme_load", enzyme_load),
    ("ggt_mcv_index", combined_alcohol_index),
]
