"""Domain features for diabetes, written from the schema and description only."""
import numpy as np
import pandas as pd


def _nz(s):
    # zeros in glucose, pressure, skin, insulin and BMI are physiologically impossible: they are missing
    return s.astype(float).replace(0, np.nan)


def glucose_x_bmi(df):
    # hyperglycaemia plus obesity is the core type-2 diabetes risk profile
    return _nz(df["plas"]) * _nz(df["mass"])


def homa_ir(df):
    # HOMA-IR insulin resistance: glucose(mg/dl) * insulin / 405
    return _nz(df["plas"]) * _nz(df["insu"]) / 405.0


def glucose_insulin_ratio(df):
    # high glucose with low insulin indicates failing beta cells
    return _nz(df["plas"]) / _nz(df["insu"])


def glucose_x_age(df):
    # glucose tolerance worsens with age; same glucose means more at a younger age
    return _nz(df["plas"]) * df["age"].astype(float)


def pedigree_x_bmi(df):
    # genetic predisposition expressed under obesity
    return df["pedi"].astype(float) * _nz(df["mass"])


def preg_per_year(df):
    # pregnancies relative to adult years; gestational diabetes risk accumulates
    return df["preg"].astype(float) / (df["age"].astype(float) - 15).clip(lower=1)


def n_missing_measures(df):
    # count of unrecorded measurements: incomplete workups differ systematically
    return (df[["plas", "pres", "skin", "insu", "mass"]].astype(float) == 0).sum(axis=1).astype(float)


def skin_to_bmi(df):
    # subcutaneous fat share of BMI (central adiposity proxy)
    return _nz(df["skin"]) / _nz(df["mass"])


PROPOSALS = [
    ("glucose_x_bmi", glucose_x_bmi),
    ("homa_ir", homa_ir),
    ("glucose_x_age", glucose_x_age),
    ("pedigree_x_bmi", pedigree_x_bmi),
    ("glucose_insulin_ratio", glucose_insulin_ratio),
    ("preg_per_year", preg_per_year),
    ("skin_to_bmi", skin_to_bmi),
    ("n_missing_measures", n_missing_measures),
]
