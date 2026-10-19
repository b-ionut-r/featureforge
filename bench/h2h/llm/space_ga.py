"""Domain features for space_ga, written from the schema and description only.

POP, EDUCATION, HOUSES, INCOME are on a log scale (per the sample values), so differences are log-ratios.
"""
import numpy as np
import pandas as pd


def edu_share(df):
    # log share of adults with 12th grade+: education strongly predicts turnout
    return df["EDUCATION"] - df["POP"]


def home_ownership(df):
    # owner-occupied houses per adult: settled residents vote more
    return df["HOUSES"] - df["POP"]


def income_per_capita(df):
    # log income per adult and per household
    return pd.DataFrame({"inc_per_pop": df["INCOME"] - df["POP"], "inc_per_house": df["INCOME"] - df["HOUSES"]})


def county_size(df):
    # absolute county population level (small rural vs metro counties) and its square
    return pd.DataFrame({"pop_sq": df["POP"] ** 2})


def coords(df):
    # rescaled coordinates and rotations so trees can find regional (South, Midwest...) turnout patterns
    x, y = df["XCOORD"] / 1e6, df["YCOORD"] / 1e6
    return pd.DataFrame({"x": x, "y": y, "x_plus_y": x + y, "x_minus_y": x - y, "r_center": np.sqrt((x + 96) ** 2 + (y - 38) ** 2)})


def edu_x_income(df):
    # combined socioeconomic status
    return (df["EDUCATION"] - df["POP"]) + (df["INCOME"] - df["POP"])


PROPOSALS = [("edu_share", edu_share), ("home_ownership", home_ownership), ("income_per_capita", income_per_capita),
             ("county_size", county_size), ("coords", coords), ("edu_x_income", edu_x_income)]
