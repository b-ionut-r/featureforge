"""Domain features for compas, written from the schema and description only."""
import numpy as np
import pandas as pd


def _n(s):
    return pd.to_numeric(s, errors="coerce").astype(float)


# Total juvenile record: any juvenile offending is an early-onset criminality signal.
def juvenile_total(df):
    j = _n(df["juv_fel_count"]) + _n(df["juv_misd_count"]) + _n(df["juv_other_count"])
    return pd.DataFrame({"juv_total": j, "any_juv": (j > 0).astype(float)}, index=df.index)


# All prior contacts (adult priors + juvenile counts) on a log scale.
def total_record(df):
    t = _n(df["priors_count"]) + _n(df["juv_fel_count"]) + _n(df["juv_misd_count"]) + _n(df["juv_other_count"])
    return np.log1p(t)


# Offending rate: priors per year of adult life; a long record at a young age signals high propensity.
def priors_per_adult_year(df):
    yrs = (_n(df["age"]) - 18).clip(lower=1)
    return _n(df["priors_count"]) / yrs


# Age-crime curve: risk drops steeply with age, so use inverse age and young flag interactions.
def age_curve(df):
    a = _n(df["age"])
    return pd.DataFrame({"inv_age": 1.0 / a.replace(0, np.nan), "young_x_priors": (a < 25).astype(float) * _n(df["priors_count"])},
                        index=df.index)


# Priors bucketed as ordinal levels (none / few / many): recidivism risk rises non-linearly with record.
def priors_level(df):
    p = _n(df["priors_count"])
    return pd.cut(p, bins=[-1, 0, 2, 5, 10, 1e9], labels=False).astype(float)


# Felony charge combined with record: a felony with priors indicates a more serious criminal trajectory.
def felony_with_priors(df):
    return _n(df["c_charge_degree_F"]) * np.log1p(_n(df["priors_count"]))


# Juvenile felonies relative to all juvenile offenses: severity of early offending.
def juv_felony_share(df):
    j = _n(df["juv_fel_count"]) + _n(df["juv_misd_count"]) + _n(df["juv_other_count"])
    return _n(df["juv_fel_count"]) / j.replace(0, np.nan)


PROPOSALS = [
    ("juvenile_total", juvenile_total),
    ("total_record", total_record),
    ("priors_per_adult_year", priors_per_adult_year),
    ("age_curve", age_curve),
    ("priors_level", priors_level),
    ("felony_with_priors", felony_with_priors),
    ("juv_felony_share", juv_felony_share),
]
