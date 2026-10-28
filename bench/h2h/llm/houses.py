"""Domain features for houses, written from the schema and description only."""
import numpy as np
import pandas as pd


def _div(a, b):
    return a / b.replace(0, np.nan)


def rooms_per_household(df):
    # average dwelling size of the block group: bigger homes cost more at equal income
    return _div(df["total_rooms"], df["households"])


def bedrooms_per_room(df):
    # low bedroom share = more living/dining space = higher-end housing stock
    return _div(df["total_bedrooms"], df["total_rooms"])


def people_per_household(df):
    # crowding: many people per household signals lower-value or rental areas
    return _div(df["population"], df["households"])


def dist_to_metros(df):
    # California prices are driven by proximity to the SF Bay and LA coastal job centres
    lat, lon = df["latitude"], df["longitude"]
    kx = np.cos(np.radians(36.0))
    out = pd.DataFrame(index=df.index)
    for name, (la, lo) in {"sf": (37.77, -122.42), "la": (34.05, -118.25),
                           "sd": (32.72, -117.16), "sj": (37.34, -121.89)}.items():
        out["dist_" + name] = np.sqrt((lat - la) ** 2 + ((lon - lo) * kx) ** 2)
    out["dist_min_metro"] = out.min(axis=1)
    return out


def rotated_coords(df):
    # the coastline runs diagonally (NW-SE); rotated axes let trees split "coastal vs inland" cleanly
    lat, lon = df["latitude"], df["longitude"]
    return pd.DataFrame({"rot45_sum": lat + lon, "rot45_diff": lat - lon,
                         "rot30": lat * np.cos(np.radians(30)) + lon * np.sin(np.radians(30))}, index=df.index)


def income_vs_neighbourhood(df):
    # a block group richer than its surrounding area (0.25 deg cell) sits in a pricier enclave
    cell = (df["latitude"] / 0.25).round().astype(str) + "_" + (df["longitude"] / 0.25).round().astype(str)
    m = df.groupby(cell)["median_income"].transform("mean")
    return pd.DataFrame({"inc_minus_cell": df["median_income"] - m, "cell_mean_income": m,
                         "cell_count": df.groupby(cell)["median_income"].transform("size").astype(float)},
                        index=df.index)


def income_per_person_room(df):
    # income relative to housing consumed: high income in small homes means expensive land
    return _div(df["median_income"], _div(df["total_rooms"], df["households"]))


PROPOSALS = [
    ("income_vs_neighbourhood", income_vs_neighbourhood),
    ("dist_to_metros", dist_to_metros),
    ("people_per_household", people_per_household),
    ("rooms_per_household", rooms_per_household),
    ("bedrooms_per_room", bedrooms_per_room),
    ("rotated_coords", rotated_coords),
    ("income_per_person_room", income_per_person_room),
]
