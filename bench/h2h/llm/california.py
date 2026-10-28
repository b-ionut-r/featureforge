"""Domain features for california, written from the schema and description only."""
import numpy as np
import pandas as pd

CITIES = {"LA": (34.05, -118.25), "SF": (37.77, -122.42), "SD": (32.72, -117.16), "SJ": (37.34, -121.89),
          "SAC": (38.58, -121.49)}


def _dist(df, lat, lon):
    la, lo = df["Latitude"].astype(float), df["Longitude"].astype(float)
    return np.sqrt((la - lat) ** 2 + ((lo - lon) * np.cos(np.radians(lat))) ** 2) * 111.0


def dist_cities(df):
    # proximity to major job centres (LA, SF Bay) drives California house prices; min = metro proximity
    d = pd.DataFrame({k: _dist(df, *v) for k, v in CITIES.items()})
    d["nearest"] = d.min(axis=1)
    return d


def rotated_coords(df):
    # the coast runs NW-SE; rotated coordinates let trees split along/perpendicular to the coastline
    la, lo = df["Latitude"].astype(float), df["Longitude"].astype(float)
    return pd.DataFrame({"r45a": la + lo, "r45b": la - lo, "r30": la * np.cos(0.52) + lo * np.sin(0.52)})


def income_vs_area(df):
    # block-group income relative to its surrounding area (~0.2 deg cell): rich enclaves vs average neighbourhoods
    cell = (df["Latitude"].astype(float) * 5).round().astype(str) + "_" + (df["Longitude"].astype(float) * 5).round().astype(str)
    m = df["MedInc"].astype(float)
    return pd.DataFrame({"area_med_inc": m.groupby(cell).transform("mean"), "inc_rel": m / m.groupby(cell).transform("mean")})


def bedroom_ratio(df):
    # share of rooms that are bedrooms: low ratio means larger living spaces / higher-end homes
    return df["AveBedrms"].astype(float) / df["AveRooms"].astype(float).replace(0, np.nan)


def households(df):
    # number of households in the block group (population / occupancy)
    return df["Population"].astype(float) / df["AveOccup"].astype(float).replace(0, np.nan)


def income_per_room(df):
    # income relative to dwelling size: affordability/pricing pressure
    return df["MedInc"].astype(float) / df["AveRooms"].astype(float).replace(0, np.nan)


def local_density(df):
    # number of block groups in the same ~0.1 deg cell: urban density proxy
    cell = (df["Latitude"].astype(float) * 10).round().astype(str) + "_" + (df["Longitude"].astype(float) * 10).round().astype(str)
    return cell.groupby(cell).transform("size").astype(float)


PROPOSALS = [("dist_cities", dist_cities), ("income_vs_area", income_vs_area), ("rotated_coords", rotated_coords),
             ("bedroom_ratio", bedroom_ratio), ("income_per_room", income_per_room), ("households", households),
             ("local_density", local_density)]
