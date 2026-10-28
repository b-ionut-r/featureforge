"""Domain features for concrete, written from the schema and description only."""
import numpy as np
import pandas as pd


def _binder(df):
    return df["cement"] + df["blast_furnace_slag"] + df["fly_ash"]


# Abrams' law: strength falls with the water-to-cement ratio.
# OUTSIDE KNOWLEDGE: Abrams' water/cement law from concrete engineering.
def water_cement_ratio(df):
    return df["water"] / df["cement"].replace(0, np.nan)


# Slag and fly ash are cementitious too; water over total binder is the modern w/b ratio.
# OUTSIDE KNOWLEDGE: water-to-binder ratio including supplementary cementitious materials.
def water_binder_ratio(df):
    return df["water"] / _binder(df).replace(0, np.nan)


# Share of the binder that is slag or fly ash (these gain strength slowly).
def scm_fraction(df):
    b = _binder(df).replace(0, np.nan)
    return (df["blast_furnace_slag"] + df["fly_ash"]) / b


# Strength grows roughly with the log of curing age.
# OUTSIDE KNOWLEDGE: logarithmic strength-maturity relation for curing age.
def log_age(df):
    return np.log1p(df["age"])


# Age interacts with w/b: hydration progress times binder richness.
def logage_x_binder_water(df):
    wb = df["water"] / _binder(df).replace(0, np.nan)
    return np.log1p(df["age"]) / wb


# Superplasticizer dose per kg of binder (lets the mix use less water).
def sp_per_binder(df):
    return df["superplasticizer"] / _binder(df).replace(0, np.nan)


# Fine to total aggregate ratio (sand ratio) affects packing and paste demand.
def sand_ratio(df):
    tot = (df["fine_aggregate"] + df["coarse_aggregate"]).replace(0, np.nan)
    return df["fine_aggregate"] / tot


# Paste share: binder plus water relative to aggregate.
def paste_to_aggregate(df):
    agg = (df["fine_aggregate"] + df["coarse_aggregate"]).replace(0, np.nan)
    return (_binder(df) + df["water"]) / agg


PROPOSALS = [
    ("water_cement_ratio", water_cement_ratio),
    ("water_binder_ratio", water_binder_ratio),
    ("scm_fraction", scm_fraction),
    ("log_age", log_age),
    ("logage_x_binder_water", logage_x_binder_water),
    ("sp_per_binder", sp_per_binder),
    ("sand_ratio", sand_ratio),
    ("paste_to_aggregate", paste_to_aggregate),
]
