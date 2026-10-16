"""Domain features for covertype, written from the schema and description only."""
import numpy as np
import pandas as pd


def elev_minus_vdist_hydro(df):
    # elevation of the nearest water table: cover types are zoned by absolute elevation and moisture
    return (df["Elevation"] - df["Vertical_Distance_To_Hydrology"]).astype(float)


def elev_minus_hdist_hydro(df):
    # elevation corrected by horizontal distance to water, a known strong linear separator between zones
    return (df["Elevation"] - 0.2 * df["Horizontal_Distance_To_Hydrology"]).astype(float)


def dist_to_hydrology(df):
    # straight-line (euclidean) distance to water
    return np.sqrt(df["Horizontal_Distance_To_Hydrology"].astype(float) ** 2
                   + df["Vertical_Distance_To_Hydrology"].astype(float) ** 2)


def hydro_fire_road_combos(df):
    # sums/differences of horizontal distances locate the cell relative to roads, fire points and water
    h = df["Horizontal_Distance_To_Hydrology"].astype(float)
    f = df["Horizontal_Distance_To_Fire_Points"].astype(float)
    r = df["Horizontal_Distance_To_Roadways"].astype(float)
    return pd.DataFrame({
        "hydro_plus_fire": h + f, "hydro_minus_fire": (h - f).abs(),
        "hydro_plus_road": h + r, "hydro_minus_road": (h - r).abs(),
        "fire_plus_road": f + r, "fire_minus_road": (f - r).abs(),
    }, index=df.index)


def soil_type_id(df):
    # collapse the 40 one-hot soil flags into one id so trees split on soil groups in one step
    cols = [c for c in df.columns if c.startswith("Soil_Type_")]
    m = df[cols].to_numpy()
    ids = np.array([int(c.split("_")[-1]) for c in cols], dtype=float)
    out = np.where(m.sum(axis=1) > 0, ids[m.argmax(axis=1)], np.nan)
    return pd.Series(out, index=df.index, dtype=float)


def wilderness_id(df):
    # collapse the 4 wilderness one-hot flags into one id
    cols = [c for c in df.columns if c.startswith("Wilderness_Area_")]
    m = df[cols].to_numpy()
    ids = np.array([int(c.split("_")[-1]) for c in cols], dtype=float)
    out = np.where(m.sum(axis=1) > 0, ids[m.argmax(axis=1)], np.nan)
    return pd.Series(out, index=df.index, dtype=float)


def aspect_sin_cos(df):
    # aspect is a circular compass angle: north/south facing slopes differ in sun and moisture
    a = np.deg2rad(df["Aspect"].astype(float))
    return pd.DataFrame({"aspect_sin": np.sin(a), "aspect_cos": np.cos(a)}, index=df.index)


def elev_vs_soil_mean(df):
    # elevation relative to the typical elevation of the same soil type
    s = soil_type_id(df)
    e = df["Elevation"].astype(float)
    return e - e.groupby(s).transform("mean")


def hillshade_stats(df):
    # total illumination and morning/afternoon asymmetry summarise terrain orientation
    a, n, p = (df[c].astype(float) for c in ["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"])
    return pd.DataFrame({"hillshade_sum": a + n + p, "hillshade_am_minus_pm": a - p}, index=df.index)


PROPOSALS = [
    ("elev_minus_vdist_hydro", elev_minus_vdist_hydro),
    ("elev_minus_hdist_hydro", elev_minus_hdist_hydro),
    ("hydro_fire_road_combos", hydro_fire_road_combos),
    ("soil_type_id", soil_type_id),
    ("elev_vs_soil_mean", elev_vs_soil_mean),
    ("dist_to_hydrology", dist_to_hydrology),
    ("wilderness_id", wilderness_id),
    ("aspect_sin_cos", aspect_sin_cos),
    ("hillshade_stats", hillshade_stats),
]
