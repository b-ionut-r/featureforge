"""Domain features for miami_housing, written from the schema and description only."""
import numpy as np
import pandas as pd


def _div(a, b):
    return a / b.replace(0, np.nan)


def _cell(df, size=0.01):
    return ((df["LATITUDE"] / size).round().astype(int).astype(str) + "_" +
            (df["LONGITUDE"] / size).round().astype(int).astype(str))


def local_market(df):
    # neighbourhood level (~1km cell): typical home size/quality nearby and this home's position in it
    c = _cell(df)
    out = pd.DataFrame(index=df.index)
    out["cell_mean_area"] = df.groupby(c)["TOT_LVG_AREA"].transform("mean")
    out["area_rel_cell"] = _div(df["TOT_LVG_AREA"], out["cell_mean_area"])
    out["cell_mean_quality"] = df.groupby(c)["structure_quality"].transform("mean")
    out["cell_mean_spec"] = df.groupby(c)["SPEC_FEAT_VAL"].transform("mean")
    out["cell_count"] = df.groupby(c)["TOT_LVG_AREA"].transform("size").astype(float)
    return out


def size_x_quality(df):
    # value ~ living area x construction quality, plus extras (pools, docks) per sq ft
    return pd.DataFrame({"area_x_quality": df["TOT_LVG_AREA"] * df["structure_quality"],
                         "spec_per_sqft": _div(df["SPEC_FEAT_VAL"], df["TOT_LVG_AREA"])}, index=df.index)


def water_access(df):
    # ocean/bay frontage dominates Miami prices: nearest water and log distances
    return pd.DataFrame({"min_water_ocean": np.minimum(df["OCEAN_DIST"], df["WATER_DIST"]),
                         "log_ocean": np.log1p(df["OCEAN_DIST"]), "log_water": np.log1p(df["WATER_DIST"]),
                         "waterfront_like": (df["WATER_DIST"] < 300).astype(float)}, index=df.index)


def lot_coverage(df):
    # land per sq ft of house: big lots in a dense city are expensive; floor-area ratio
    return pd.DataFrame({"land_per_living": _div(df["LND_SQFOOT"], df["TOT_LVG_AREA"]),
                         "log_land": np.log1p(df["LND_SQFOOT"])}, index=df.index)


def accessibility(df):
    # access to centres: CBD vs sub-centre, and highway nuisance vs access
    return pd.DataFrame({"min_centre": np.minimum(df["CNTR_DIST"], df["SUBCNTR_DI"]),
                         "cntr_minus_ocean": df["CNTR_DIST"] - df["OCEAN_DIST"],
                         "log_hwy": np.log1p(df["HWY_DIST"]), "log_rail": np.log1p(df["RAIL_DIST"])}, index=df.index)


def rotated_coords(df):
    # rotated coordinates follow the N-S coastline of Biscayne Bay
    return pd.DataFrame({"rot_sum": df["LATITUDE"] + df["LONGITUDE"],
                         "rot_diff": df["LATITUDE"] - df["LONGITUDE"]}, index=df.index)


PROPOSALS = [
    ("size_x_quality", size_x_quality),
    ("local_market", local_market),
    ("water_access", water_access),
    ("lot_coverage", lot_coverage),
    ("accessibility", accessibility),
    ("rotated_coords", rotated_coords),
]
