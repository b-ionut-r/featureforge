"""Domain features for quake, written from the schema and description only.

Only depth, latitude and longitude are available; features are geophysical transforms of location and depth.
"""
import numpy as np
import pandas as pd


def log_depth(df):
    # depths are heavily right-skewed (most quakes shallow)
    return np.log1p(df["focal_depth"].clip(lower=0))


def depth_class(df):
    # OUTSIDE KNOWLEDGE: seismological classes shallow <70 km, intermediate 70-300 km, deep >300 km
    d = df["focal_depth"]
    return pd.DataFrame({"intermediate": ((d >= 70) & (d < 300)).astype(float), "deep": (d >= 300).astype(float),
                         "depth_cls": np.select([d < 70, d < 300], [0, 1], 2).astype(float)})


def sphere_xyz(df):
    # 3-D unit-sphere coordinates: removes longitude wrap-around at +/-180
    lat, lon = np.radians(df["latitude"]), np.radians(df["longitude"])
    return pd.DataFrame({"x": np.cos(lat) * np.cos(lon), "y": np.cos(lat) * np.sin(lon), "z": np.sin(lat)})


def abs_lat(df):
    # distance from equator
    return df["latitude"].abs()


def pacific_rim(df):
    # OUTSIDE KNOWLEDGE: Pacific "Ring of Fire" subduction zones (west Pacific lon>100, Americas lon<-60)
    lon = df["longitude"]
    return ((lon > 100) | (lon < -60)).astype(float)


def depth_x_location(df):
    # deep-focus events cluster at specific slabs (Tonga/Fiji, Japan); depth interactions with location
    ld = log_depth(df)
    return pd.DataFrame({"ld_x_lat": ld * df["latitude"], "ld_x_coslon": ld * np.cos(np.radians(df["longitude"])),
                         "ld_x_sinlon": ld * np.sin(np.radians(df["longitude"]))})


PROPOSALS = [("log_depth", log_depth), ("depth_class", depth_class), ("sphere_xyz", sphere_xyz),
             ("abs_lat", abs_lat), ("pacific_rim", pacific_rim), ("depth_x_location", depth_x_location)]
