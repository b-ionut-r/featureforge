"""Domain features for hr_analytics, written from the schema and description only."""
import numpy as np
import pandas as pd


def _s(df, c):
    return df[c].astype(object).where(df[c].notna(), "nan").astype(str)


def _experience(df):
    s = _s(df, "experience").replace({">20": "21", "<1": "0", "nan": np.nan})
    return pd.to_numeric(s, errors="coerce")


def _last_new_job(df):
    s = _s(df, "last_new_job").replace({">4": "5", "never": "0", "nan": np.nan})
    return pd.to_numeric(s, errors="coerce")


def _company_size(df):
    m = {"<10": 5, "10/49": 30, "10-49": 30, "50-99": 75, "100-500": 300, "500-999": 750,
         "1000-4999": 3000, "5000-9999": 7500, "10000+": 15000}
    return _s(df, "company_size").map(m).astype(float)


def missing_employer_info(df):
    # candidates with no company size/type are often not employed or hiding employer: strong job-seeker signal
    cols = ["gender", "enrolled_university", "education_level", "major_discipline", "company_size", "company_type"]
    miss = sum(_s(df, c).isin(["nan", "None", ""]).astype(float) for c in cols)
    emp = (_s(df, "company_size").isin(["nan", "None", ""]) & _s(df, "company_type").isin(["nan", "None", ""]))
    return pd.DataFrame({"n_missing": miss, "no_employer": emp.astype(float)}, index=df.index)


def experience_years(df):
    # years of experience as an ordered number: juniors switch jobs much more often
    return _experience(df)


def years_since_job_change(df):
    # years since last job change, numeric (never=0, >4=5): short tenures signal frequent movers
    return _last_new_job(df)


def experience_minus_tenure(df):
    # experience accumulated before the current job; career changers vs. long-stayers
    return _experience(df) - _last_new_job(df)


def company_size_num(df):
    # employer headcount midpoint: people in small firms/startups are more mobile
    return np.log1p(_company_size(df))


def city_frequency(df):
    # number of candidates from the same city: hub cities have many alternative employers
    c = _s(df, "city")
    return c.groupby(c).transform("size").astype(float)


def training_per_experience(df):
    # training hours relative to experience: heavy upskilling by juniors suggests preparing a move
    return df["training_hours"] / (_experience(df) + 1)


def cdi_vs_city_rank(df):
    # low-development cities push talent to look for jobs elsewhere; interaction with experience
    return (1 - df["city_development_index"]) * (21 - _experience(df).fillna(10))


PROPOSALS = [
    ("missing_employer_info", missing_employer_info),
    ("cdi_vs_city_rank", cdi_vs_city_rank),
    ("experience_years", experience_years),
    ("years_since_job_change", years_since_job_change),
    ("experience_minus_tenure", experience_minus_tenure),
    ("company_size_num", company_size_num),
    ("city_frequency", city_frequency),
    ("training_per_experience", training_per_experience),
]
