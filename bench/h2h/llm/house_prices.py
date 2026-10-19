"""Domain features for house_prices, written from the schema and description only."""
import numpy as np
import pandas as pd

QUAL = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}


def _q(df, c):
    return df[c].map(QUAL).astype(float)


def _n(df, c):
    return pd.to_numeric(df[c], errors="coerce").astype(float)


def total_sf(df):
    # total finished living space incl. basement: buyers pay per square foot of usable area
    return _n(df, "TotalBsmtSF").fillna(0) + _n(df, "1stFlrSF") + _n(df, "2ndFlrSF")


def qual_x_area(df):
    # overall quality times total area: the dominant interaction in Ames prices
    return _n(df, "OverallQual") * total_sf(df)


def total_baths(df):
    # bathrooms counted with half baths as 0.5
    return (_n(df, "FullBath") + 0.5 * _n(df, "HalfBath")
            + _n(df, "BsmtFullBath").fillna(0) + 0.5 * _n(df, "BsmtHalfBath").fillna(0))


def ages(df):
    # age at sale and years since remodel: depreciation
    return pd.DataFrame({"age_at_sale": _n(df, "YrSold") - _n(df, "YearBuilt"),
                         "years_since_remod": _n(df, "YrSold") - _n(df, "YearRemodAdd")}, index=df.index)


def quality_score(df):
    # sum of ordinal quality grades for exterior, kitchen, basement, heating, fireplace, garage
    cols = ["ExterQual", "KitchenQual", "BsmtQual", "HeatingQC", "FireplaceQu", "GarageQual"]
    return sum(_q(df, c).fillna(0) for c in cols if c in df.columns)


def area_vs_neighborhood(df):
    # house size relative to its neighborhood: larger than neighbours sells at a premium
    a = total_sf(df)
    return a / a.groupby(df["Neighborhood"].astype(str)).transform("median")


def qual_vs_neighborhood(df):
    # quality relative to the neighborhood mean
    q = _n(df, "OverallQual")
    return q - q.groupby(df["Neighborhood"].astype(str)).transform("mean")


def total_porch_sf(df):
    # all outdoor living space
    cols = ["WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]
    return sum(_n(df, c).fillna(0) for c in cols)


def lot_per_living(df):
    # lot area per living area: land share of the value
    return _n(df, "LotArea") / _n(df, "GrLivArea").replace(0, np.nan)


def neighborhood_quality_level(df):
    # neighborhood's mean overall quality: location premium proxy without using the target
    q = _n(df, "OverallQual")
    return q.groupby(df["Neighborhood"].astype(str)).transform("mean")


PROPOSALS = [
    ("qual_x_area", qual_x_area),
    ("total_sf", total_sf),
    ("total_baths", total_baths),
    ("ages", ages),
    ("quality_score", quality_score),
    ("neighborhood_quality_level", neighborhood_quality_level),
    ("area_vs_neighborhood", area_vs_neighborhood),
    ("qual_vs_neighborhood", qual_vs_neighborhood),
    ("total_porch_sf", total_porch_sf),
    ("lot_per_living", lot_per_living),
]
