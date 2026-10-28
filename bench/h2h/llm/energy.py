"""Domain features for energy, written from the schema and description only."""
import numpy as np
import pandas as pd


# Building volume: footprint (roof area) times height; drives heat capacity and load.
def volume(df):
    return df["roof_area"] * df["overall_height"]


# Envelope surface per unit volume: heat loss scales with exposed surface.
def surface_to_volume(df):
    v = (df["roof_area"] * df["overall_height"]).replace(0, np.nan)
    return df["surface_area"] / v


# Absolute glazed area: glazing fraction times wall area (windows lose/gain most heat).
def glazed_area(df):
    return df["glazing_area"] * df["wall_area"]


# Roof share of envelope: roofs see solar gain and lose heat upward.
def roof_share(df):
    return df["roof_area"] / df["surface_area"].replace(0, np.nan)


# Wall share of envelope.
def wall_share(df):
    return df["wall_area"] / df["surface_area"].replace(0, np.nan)


# Tall (two-storey) building flag; height has only two levels.
def is_tall(df):
    return (df["overall_height"] > 5).astype(float)


# Glazing present and whether distributed uniformly vs concentrated on one side.
# OUTSIDE KNOWLEDGE: in the Tsanas & Xifara data, distribution 0 = no glazing, 1 = uniform, 2-5 = one facade.
def glazing_distribution_flags(df):
    d = df["glazing_area_distribution"].astype(float)
    return pd.DataFrame({
        "has_glazing": (df["glazing_area"] > 0).astype(float),
        "glaz_uniform": (d == 1).astype(float),
        "glaz_one_side": (d >= 2).astype(float),
    }, index=df.index)


# Compactness times height: compact, tall buildings behave very differently from low spread ones.
def compactness_x_height(df):
    return df["relative_compactness"] * df["overall_height"]


PROPOSALS = [
    ("volume", volume),
    ("surface_to_volume", surface_to_volume),
    ("glazed_area", glazed_area),
    ("roof_share", roof_share),
    ("wall_share", wall_share),
    ("is_tall", is_tall),
    ("glazing_distribution_flags", glazing_distribution_flags),
    ("compactness_x_height", compactness_x_height),
]
