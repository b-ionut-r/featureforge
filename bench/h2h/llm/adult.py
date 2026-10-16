"""Domain features for adult, written from the schema and description only."""
import numpy as np
import pandas as pd


def _s(df, c):
    return df[c].astype("string").str.strip()


def capital_net(df):
    # net investment income: any capital gain signals wealth, which strongly separates >50K earners
    return df["capital-gain"].astype(float) - df["capital-loss"].astype(float)


def is_married_spouse(df):
    # married with spouse present = dual/household income reported, the single strongest income marker
    ms = _s(df, "marital-status")
    return ms.isin(["Married-civ-spouse", "Married-AF-spouse"]).astype(float)


def edu_x_hours(df):
    # human capital times labour supply approximates earning power
    return df["education-num"].astype(float) * df["hours-per-week"].astype(float)


def edu_vs_occupation(df):
    # being more educated than typical for one's occupation suggests seniority / higher pay grade
    occ = _s(df, "occupation").fillna("na")
    e = df["education-num"].astype(float)
    return e - e.groupby(occ).transform("mean")


def hours_vs_occupation(df):
    # working more hours than peers in the same occupation implies salaried/managerial or overtime pay
    occ = _s(df, "occupation").fillna("na")
    h = df["hours-per-week"].astype(float)
    return h - h.groupby(occ).transform("mean")


def age_experience(df):
    # potential labour market experience (Mincer): age minus schooling years minus 6
    return (df["age"].astype(float) - df["education-num"].astype(float) - 6).clip(lower=0)


def log_capital_gain_per_hour(df):
    # capital gain scaled by work intensity separates investors from wage workers
    return np.log1p(df["capital-gain"].astype(float).clip(lower=0)) / df["hours-per-week"].astype(float).replace(0, np.nan)


PROPOSALS = [("capital_net", capital_net), ("is_married_spouse", is_married_spouse), ("edu_x_hours", edu_x_hours),
             ("age_experience", age_experience), ("edu_vs_occupation", edu_vs_occupation),
             ("hours_vs_occupation", hours_vs_occupation), ("log_capital_gain_per_hour", log_capital_gain_per_hour)]
