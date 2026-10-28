"""Domain features for kc_house, written from the schema and description only."""
import numpy as np
import pandas as pd


def _div(a, b):
    return a / b.replace(0, np.nan)


def living_vs_zip(df):
    # a home's size and grade relative to its zipcode: price ~ size x location price level
    z = df["zipcode"].astype(str)
    out = pd.DataFrame(index=df.index)
    out["sqft_rel_zip"] = _div(df["sqft_living"], df.groupby(z)["sqft_living"].transform("mean"))
    out["grade_minus_zip"] = df["grade"] - df.groupby(z)["grade"].transform("mean")
    out["zip_mean_grade"] = df.groupby(z)["grade"].transform("mean")
    out["zip_mean_sqft"] = df.groupby(z)["sqft_living"].transform("mean")
    return out


def dist_to_seattle(df):
    # distance to downtown Seattle and Bellevue job centres drives land value
    kx = np.cos(np.radians(47.6))
    d1 = np.sqrt((df["lat"] - 47.6062) ** 2 + ((df["long"] + 122.3321) * kx) ** 2)
    d2 = np.sqrt((df["lat"] - 47.6101) ** 2 + ((df["long"] + 122.2015) * kx) ** 2)
    return pd.DataFrame({"dist_seattle_km": d1 * 111, "dist_bellevue_km": d2 * 111,
                         "dist_min_km": np.minimum(d1, d2) * 111}, index=df.index)


def size_x_grade(df):
    # value ~ size times quality; trees can't multiply, so give them the product
    return np.log1p(df["sqft_living"]) * df["grade"]


def living_vs_neighbours(df):
    # bigger than the 15 nearest neighbours = top house on the street (and vice versa)
    return pd.DataFrame({"liv_ratio15": _div(df["sqft_living"], df["sqft_living15"]),
                         "lot_ratio15": _div(df["sqft_lot"], df["sqft_lot15"])}, index=df.index)


def age_at_sale(df):
    # age and years since last renovation at sale time
    age = df["date_year"] - df["yr_built"]
    last = np.where(df["yr_renovated"] > 0, df["yr_renovated"], df["yr_built"])
    return pd.DataFrame({"age_at_sale": age.astype(float), "yrs_since_reno": (df["date_year"] - last).astype(float)},
                        index=df.index)


def sale_time(df):
    # months since May 2014: captures the rising 2014-15 market trend
    return ((df["date_year"] - 2014) * 12 + df["date_month"] - 5 + df["date_day"] / 31.0).astype(float)


def layout_ratios(df):
    # bathrooms per bedroom, basement share, floor-area ratio: layout quality signals
    return pd.DataFrame({"bath_per_bed": _div(df["bathrooms"], df["bedrooms"].astype(float)),
                         "basement_share": _div(df["sqft_basement"].astype(float), df["sqft_living"].astype(float)),
                         "floor_area_ratio": _div(df["sqft_living"].astype(float), df["sqft_lot"].astype(float)),
                         "sqft_per_room": _div(df["sqft_living"].astype(float),
                                               (df["bedrooms"] + df["bathrooms"]).astype(float))}, index=df.index)


def rotated_coords(df):
    # rotated lat/long so trees can carve diagonal neighbourhood boundaries (Lake Washington, Sound)
    return pd.DataFrame({"rot_sum": df["lat"] + df["long"], "rot_diff": df["lat"] - df["long"]}, index=df.index)


PROPOSALS = [
    ("living_vs_zip", living_vs_zip),
    ("size_x_grade", size_x_grade),
    ("dist_to_seattle", dist_to_seattle),
    ("living_vs_neighbours", living_vs_neighbours),
    ("age_at_sale", age_at_sale),
    ("layout_ratios", layout_ratios),
    ("sale_time", sale_time),
    ("rotated_coords", rotated_coords),
]
