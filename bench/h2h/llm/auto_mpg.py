"""Domain features for auto_mpg, written from the schema and description only."""
import numpy as np
import pandas as pd


def _f(df, c):
    return pd.to_numeric(df[c].astype(object), errors="coerce").astype(float)


def power_to_weight(df):
    # horsepower per unit weight: performance orientation; fuel use rises with power and mass
    return pd.DataFrame({"hp_per_ton": df["horsepower"] / df["weight"].replace(0, np.nan) * 2000,
                         "weight_per_hp": df["weight"] / df["horsepower"].replace(0, np.nan)})


def displacement_ratios(df):
    # engine size per cylinder and per weight: big-bore engines and oversized engines burn more fuel
    cyl = _f(df, "cylinders").replace(0, np.nan)
    return pd.DataFrame({"disp_per_cyl": df["displacement"] / cyl, "disp_per_weight": df["displacement"] / df["weight"].replace(0, np.nan),
                         "hp_per_disp": df["horsepower"] / df["displacement"].replace(0, np.nan)})


def inverse_physics(df):
    # fuel consumption (gallons/mile) is roughly linear in weight and displacement, so mpg ~ 1/weight
    return pd.DataFrame({"inv_weight": 1000.0 / df["weight"].replace(0, np.nan),
                         "inv_disp": 100.0 / df["displacement"].replace(0, np.nan),
                         "log_weight": np.log(df["weight"].clip(lower=1))})


def model_year(df):
    # model year (70-82): post-1973 oil crisis and CAFE standards drove efficiency up over time
    # OUTSIDE KNOWLEDGE: 1973/1979 oil shocks and 1978 CAFE standards
    y = _f(df, "model")
    return pd.DataFrame({"year": y, "post_oil_crisis": (y >= 74).astype(float), "post_cafe": (y >= 78).astype(float)})


def origin_flags(df):
    # origin codes 1/2/3 = USA/Europe/Japan: imports were small, efficient cars
    # OUTSIDE KNOWLEDGE: auto-mpg origin coding 1=USA, 2=Europe, 3=Japan
    o = df["origin"].astype(object).astype(str)
    return pd.DataFrame({"is_usa": (o == "1").astype(float), "is_japan": (o == "3").astype(float)})


def accel_power(df):
    # acceleration (0-60 time) times power/weight: tuning for performance vs economy
    return pd.DataFrame({"accel_x_hppw": df["acceleration"] * df["horsepower"] / df["weight"].replace(0, np.nan),
                         "weight_x_accel": df["weight"] / df["acceleration"].replace(0, np.nan)})


def year_x_weight(df):
    # newer technology at a given weight: year-adjusted mass
    return _f(df, "model") * 1000.0 / df["weight"].replace(0, np.nan)


PROPOSALS = [("power_to_weight", power_to_weight), ("displacement_ratios", displacement_ratios),
             ("inverse_physics", inverse_physics), ("model_year", model_year), ("origin_flags", origin_flags),
             ("accel_power", accel_power), ("year_x_weight", year_x_weight)]
