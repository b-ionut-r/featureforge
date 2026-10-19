"""Domain features for data, written from the schema and description only."""
import numpy as np
import pandas as pd

# OUTSIDE KNOWLEDGE: 768 rows with these levels match the building energy-efficiency simulation table:
# f_1 relative compactness, f_2 surface area, f_3 wall area, f_4 roof area, f_5 overall height,
# f_6 orientation (2-5 = N,E,S,W), f_7 glazing area fraction, f_8 glazing distribution
# (0 none, 1 uniform, 2-5 = N,E,S,W). Target is heating or cooling load.


def envelope_shares(df):
    # heat loss splits across walls/roof/floor; shares of the envelope matter for transmission losses
    s = df["f_2"].replace(0, np.nan)
    floor = df["f_2"] - df["f_3"] - df["f_4"]
    return pd.DataFrame({
        "wall_share": df["f_3"] / s,
        "roof_share": df["f_4"] / s,
        "floor_area_resid": floor,
    })


def glazing_amount(df):
    # absolute window area (glazing fraction of floor area) - windows dominate conductive and solar gains
    floors = (df["f_5"] / 3.5).round()
    floor_area = df["f_4"] * floors
    gl = df["f_7"] * floor_area
    return pd.DataFrame({
        "glazing_m2": gl,
        "glazing_over_wall": gl / df["f_3"].replace(0, np.nan),
        "n_floors": floors,
    })


def glazing_facing(df):
    # actual compass direction of concentrated glazing = distribution side rotated by building orientation
    d = df["f_8"]
    o = df["f_6"]
    facing = ((d - 2) + (o - 2)) % 4  # 0=N,1=E,2=S,3=W
    concentrated = d >= 2
    out = {
        "glz_uniform": (d == 1).astype(float) * df["f_7"],
        "glz_none": (df["f_7"] <= 0).astype(float),
    }
    for k, name in enumerate(["n", "e", "s", "w"]):
        out["glz_face_" + name] = ((facing == k) & concentrated).astype(float) * df["f_7"]
    return pd.DataFrame(out)


def compactness_height(df):
    # tall compact buildings have less envelope per volume; interaction with glazing
    return pd.DataFrame({
        "tall": (df["f_5"] > 5).astype(float),
        "compact_x_height": df["f_1"] * df["f_5"],
        "surface_x_glazing": df["f_2"] * df["f_7"],
        "roof_per_height": df["f_4"] / df["f_5"].replace(0, np.nan),
    })


def ua_proxy(df):
    # transmission loss proxy: walls+roof weighted, windows much worse U-value than walls
    # OUTSIDE KNOWLEDGE: typical U-values wall ~0.5, roof ~0.5, floor ~0.9, glazing ~2.3 W/m2K in that simulation
    floor = (df["f_2"] - df["f_3"] - df["f_4"]).clip(lower=0)
    floors = (df["f_5"] / 3.5).round()
    gl = df["f_7"] * df["f_4"] * floors
    ua = 0.5 * (df["f_3"] - gl).clip(lower=0) + 0.5 * df["f_4"] + 0.9 * floor + 2.3 * gl
    return ua.rename("ua_proxy")


PROPOSALS = [
    ("envelope_shares", envelope_shares),
    ("glazing_amount", glazing_amount),
    ("glazing_facing", glazing_facing),
    ("compactness_height", compactness_height),
    ("ua_proxy", ua_proxy),
]
