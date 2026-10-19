"""Domain features for boston, written from the schema and description only."""
import numpy as np
import pandas as pd


def _n(df, c):
    return pd.to_numeric(df[c], errors="coerce").astype(float)


def rooms_sq(df):
    # hedonic price models (Harrison & Rubinfeld) use RM^2: big houses command a convex premium
    return _n(df, "RM") ** 2


def log_transforms(df):
    # crime, lower-status share and distance act multiplicatively on price
    return pd.DataFrame({"log_crim": np.log(_n(df, "CRIM").clip(lower=1e-4)), "log_lstat": np.log(_n(df, "LSTAT").clip(lower=0.1)),
                         "log_dis": np.log(_n(df, "DIS").clip(lower=0.1))})


def tax_per_rad(df):
    # tax burden relative to highway accessibility (urban services vs cost)
    return _n(df, "TAX") / _n(df, "RAD").replace(0, np.nan)


def pollution_industry(df):
    # NOX is driven by industry; NOX^2 as in the clean-air hedonic model, and NOX per industrial share
    return pd.DataFrame({"nox_sq": _n(df, "NOX") ** 2, "nox_x_indus": _n(df, "NOX") * _n(df, "INDUS")})


def rooms_per_status(df):
    # space per socioeconomic disadvantage: large homes in affluent areas
    return pd.DataFrame({"rm_per_lstat": _n(df, "RM") / _n(df, "LSTAT").replace(0, np.nan),
                         "rm_x_lstat": _n(df, "RM") * _n(df, "LSTAT")})


def bk_decode(df):
    # invert B = 1000(Bk-0.63)^2 on the Bk<0.63 branch to recover the actual proportion
    return 0.63 - np.sqrt((_n(df, "B") / 1000).clip(lower=0))


def old_and_far(df):
    # old housing stock near employment centres = dense older inner city
    return pd.DataFrame({"age_per_dis": _n(df, "AGE") / _n(df, "DIS").replace(0, np.nan),
                         "zn_positive": (_n(df, "ZN") > 0).astype(float)})


def school_quality(df):
    # pupil-teacher ratio with status: schooling quality for families
    return _n(df, "PTRATIO") * _n(df, "LSTAT")


PROPOSALS = [("rooms_sq", rooms_sq), ("log_transforms", log_transforms), ("tax_per_rad", tax_per_rad),
             ("pollution_industry", pollution_industry), ("rooms_per_status", rooms_per_status),
             ("bk_decode", bk_decode), ("old_and_far", old_and_far), ("school_quality", school_quality)]
