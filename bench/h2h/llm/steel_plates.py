"""Domain features for steel_plates, written from the schema and description only."""
import numpy as np
import pandas as pd


def bbox(df):
    # bounding-box width/height/aspect of the defect: scratches are long and thin, stains compact
    w = (df["X_Maximum"] - df["X_Minimum"]).astype(float)
    h = (df["Y_Maximum"] - df["Y_Minimum"]).astype(float)
    return pd.DataFrame({"width": w, "height": h, "log_aspect": np.log((w + 1) / (h + 1))})


def fill_ratio(df):
    # defect pixels over bounding-box area: solid blobs vs sparse/diagonal marks
    w = (df["X_Maximum"] - df["X_Minimum"] + 1).astype(float)
    h = (df["Y_Maximum"] - df["Y_Minimum"] + 1).astype(float)
    return df["Pixels_Areas"] / (w * h)


def mean_luminosity(df):
    # average brightness of defect pixels (dirt/stains darker, bumps/scratches brighter)
    return df["Sum_of_Luminosity"] / df["Pixels_Areas"].replace(0, np.nan)


def lum_contrast(df):
    # luminosity range and where the mean sits within it
    rng = df["Maximum_of_Luminosity"].astype(float) - df["Minimum_of_Luminosity"].astype(float)
    mean = df["Sum_of_Luminosity"] / df["Pixels_Areas"].replace(0, np.nan)
    return pd.DataFrame({"lum_range": rng,
                         "lum_pos": (mean - df["Minimum_of_Luminosity"]) / rng.replace(0, np.nan)})


def compactness(df):
    # perimeter^2 / area: irregular, jagged outlines vs round blobs
    p = (df["X_Perimeter"] + df["Y_Perimeter"]).astype(float)
    return p ** 2 / df["Pixels_Areas"].replace(0, np.nan)


def perimeter_ratio(df):
    # X vs Y perimeter balance: orientation of elongated faults
    return np.log((df["X_Perimeter"] + 1) / (df["Y_Perimeter"] + 1))


def x_position(df):
    # lateral position across the plate: edge-located faults (e.g. Z/K scratches) vs center
    xc = (df["X_Minimum"] + df["X_Maximum"]) / 2.0
    pos = xc / df["Length_of_Conveyer"].replace(0, np.nan)
    return pd.DataFrame({"x_rel": pos, "edge_dist": np.minimum(pos, 1 - pos)})


def area_per_thickness(df):
    # defect size relative to plate thickness
    return df["Pixels_Areas"] / df["Steel_Plate_Thickness"].replace(0, np.nan)


PROPOSALS = [("bbox", bbox), ("mean_luminosity", mean_luminosity), ("fill_ratio", fill_ratio), ("lum_contrast", lum_contrast),
             ("x_position", x_position), ("compactness", compactness), ("perimeter_ratio", perimeter_ratio),
             ("area_per_thickness", area_per_thickness)]
