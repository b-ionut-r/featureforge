"""Domain features for data, written from the schema and description only."""
import numpy as np
import pandas as pd

# OUTSIDE KNOWLEDGE: 5278 rows with a binary, an age, three small juvenile-like counts, a priors-like count,
# three one-hot age bands (sample rows: age 22 -> f_9, 40 -> f_7, 51 -> f_8), two race one-hots and two
# charge-degree one-hots matches the COMPAS two-year recidivism table: f_1 sex, f_2 age, f_3 juv_fel_count,
# f_4 juv_misd_count, f_5 juv_other_count, f_6 priors_count, f_7..f_9 age categories, f_10/f_11 race,
# f_12/f_13 charge degree.


def _num(s):
    return pd.to_numeric(s, errors="coerce").astype(float)


def juvenile_history(df):
    # total juvenile record and whether any exists - early offending predicts reoffending
    j = _num(df["f_3"]) + _num(df["f_4"]) + _num(df["f_5"])
    return pd.DataFrame({"juv_total": j, "juv_any": (j > 0).astype(float),
                         "juv_fel_any": (_num(df["f_3"]) > 0).astype(float)})


def priors_transforms(df):
    # prior-offense count is heavy-tailed; log and thresholds
    p = _num(df["f_6"]).clip(lower=0)
    return pd.DataFrame({"log_priors": np.log1p(p), "priors_ge3": (p >= 3).astype(float),
                         "priors_zero": (p == 0).astype(float)})


def offense_rate(df):
    # criminal-history intensity: priors per year of adult exposure (age - 18)
    age = _num(df["f_2"])
    yrs = (age - 17).clip(lower=1)
    p = _num(df["f_6"]).clip(lower=0)
    j = _num(df["f_3"]) + _num(df["f_4"]) + _num(df["f_5"])
    return pd.DataFrame({"priors_per_adult_year": p / yrs, "all_per_year": (p + j) / yrs})


def age_curve(df):
    # age-crime curve: reoffending risk falls steeply with age
    age = _num(df["f_2"])
    return pd.DataFrame({"inv_age": 1.0 / age.replace(0, np.nan), "log_age": np.log(age.clip(lower=1))})


def history_x_age(df):
    # young people with long records are the highest-risk group
    age = _num(df["f_2"])
    p = _num(df["f_6"]).clip(lower=0)
    return pd.DataFrame({"logpriors_over_age": np.log1p(p) / age.replace(0, np.nan) * 30.0,
                         "young_with_priors": ((age < 25) & (p > 0)).astype(float)})


def charge_x_history(df):
    # current charge degree interacting with record length
    p = np.log1p(_num(df["f_6"]).clip(lower=0))
    return pd.DataFrame({"charge12_x_priors": _num(df["f_12"]) * p,
                         "charge13_x_priors": _num(df["f_13"]) * p})


PROPOSALS = [
    ("juvenile_history", juvenile_history),
    ("priors_transforms", priors_transforms),
    ("offense_rate", offense_rate),
    ("age_curve", age_curve),
    ("history_x_age", history_x_age),
    ("charge_x_history", charge_x_history),
]
