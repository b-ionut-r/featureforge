"""Domain features for wind, written from the schema and description only.

Daily mean wind speeds (knots) at 12 Irish synoptic stations 1961-78; the 12th station (Malin Head,
Donegal) is the usual target, so features combine the other stations, weighting the northern/western ones."""
import numpy as np
import pandas as pd

ST = ["RPT", "VAL", "ROS", "KIL", "SHA", "BIR", "DUB", "CLA", "MUL", "CLO", "BEL"]


def station_mean(df):
    # national mean wind: the synoptic-scale weather system dominates every station
    return df[ST].mean(axis=1)


def north_west_mean(df):
    # stations nearest the north coast (Belmullet, Clones, Claremorris, Mullingar) track Malin Head best
    return df[["BEL", "CLO", "CLA", "MUL"]].mean(axis=1)


def coastal_inland_ratio(df):
    # coastal (Belmullet, Valentia, Roche's Pt, Rosslare) vs inland means: proxy for gradient strength
    coast = df[["BEL", "VAL", "RPT", "ROS"]].mean(axis=1)
    inland = df[["BIR", "MUL", "CLO", "KIL", "CLA"]].mean(axis=1)
    return coast / inland.replace(0, np.nan)


def west_east_diff(df):
    # west-east gradient: Atlantic depressions approach from the west
    return df[["BEL", "VAL", "SHA", "CLA"]].mean(axis=1) - df[["DUB", "RPT", "ROS", "KIL"]].mean(axis=1)


def station_spread(df):
    # spatial variability: a local front/gale vs uniform calm
    return pd.DataFrame({"st_std": df[ST].std(axis=1), "st_max": df[ST].max(axis=1), "st_median": df[ST].median(axis=1)})


def season(df):
    # winter months are windier: annual cycle as sin/cos of day of year
    doy = (pd.to_numeric(df["month"], errors="coerce") - 1) * 30.44 + pd.to_numeric(df["day"], errors="coerce")
    return pd.DataFrame({"doy_sin": np.sin(2 * np.pi * doy / 365.25), "doy_cos": np.cos(2 * np.pi * doy / 365.25)})


def bel_rel_month_mean(df):
    # Belmullet anomaly vs its (year, month) average: storm days stand out from the climatology
    key = df["year"].astype(str) + "-" + df["month"].astype(str)
    return df["BEL"] - df["BEL"].groupby(key).transform("mean")


PROPOSALS = [("north_west_mean", north_west_mean), ("station_mean", station_mean), ("station_spread", station_spread),
             ("west_east_diff", west_east_diff), ("coastal_inland_ratio", coastal_inland_ratio), ("season", season),
             ("bel_rel_month_mean", bel_rel_month_mean)]
