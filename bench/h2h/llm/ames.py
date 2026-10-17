"""Domain features for ames, written from the schema and description only."""
import numpy as np
import pandas as pd

QMAP = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}


def _q(df, c):
    return df[c].astype("string").str.strip().map(QMAP).astype(float)


def _n(df, c):
    return pd.to_numeric(df[c], errors="coerce").astype(float)


def total_sf(df):
    # total finished+basement floor area is the dominant driver of house price
    return _n(df, "Total.Bsmt.SF").fillna(0) + _n(df, "X1st.Flr.SF") + _n(df, "X2nd.Flr.SF")


def qual_x_area(df):
    # price ~ quality-adjusted living area: quality multiplies the value of each square foot
    return _n(df, "Overall.Qual") * _n(df, "Gr.Liv.Area")


def area_vs_neighborhood(df):
    # living area relative to the neighborhood's median: largest house on the block vs typical
    a = _n(df, "Gr.Liv.Area")
    return a / a.groupby(df["Neighborhood"].astype(str)).transform("median")


def age_at_sale(df):
    # depreciation: years from construction and from last remodel to the sale
    return pd.DataFrame({"age": _n(df, "Yr.Sold") - _n(df, "Year.Built"),
                         "since_remod": _n(df, "Yr.Sold") - _n(df, "Year.Remod.Add")})


def total_baths(df):
    # bathrooms (half counting 0.5) are a key amenity in valuation
    return (_n(df, "Full.Bath") + 0.5 * _n(df, "Half.Bath") + _n(df, "Bsmt.Full.Bath").fillna(0)
            + 0.5 * _n(df, "Bsmt.Half.Bath").fillna(0))


def quality_score(df):
    # summed ordinal quality ratings of exterior, kitchen, basement, heating, fireplace, garage
    cols = ["Exter.Qual", "Kitchen.Qual", "Bsmt.Qual", "Heating.QC", "Fireplace.Qu", "Garage.Qual"]
    return pd.concat([_q(df, c) for c in cols], axis=1).fillna(0).sum(axis=1)


def total_porch(df):
    # outdoor living space adds value
    return sum(_n(df, c) for c in ["Wood.Deck.SF", "Open.Porch.SF", "Enclosed.Porch", "X3Ssn.Porch", "Screen.Porch"])


def neighborhood_quality(df):
    # overall quality relative to neighborhood: a nicer-than-neighbors house commands a premium
    q = _n(df, "Overall.Qual")
    return q - q.groupby(df["Neighborhood"].astype(str)).transform("mean")


def lot_vs_neighborhood(df):
    # lot size relative to neighborhood median (big lots matter within a location)
    a = _n(df, "Lot.Area")
    return a / a.groupby(df["Neighborhood"].astype(str)).transform("median")


def qual_total_sf(df):
    # quality times total area including basement (with basement quality)
    return _n(df, "Overall.Qual") * total_sf(df)


PROPOSALS = [("qual_total_sf", qual_total_sf), ("total_sf", total_sf), ("qual_x_area", qual_x_area),
             ("age_at_sale", age_at_sale), ("total_baths", total_baths), ("quality_score", quality_score),
             ("area_vs_neighborhood", area_vs_neighborhood), ("neighborhood_quality", neighborhood_quality),
             ("total_porch", total_porch), ("lot_vs_neighborhood", lot_vs_neighborhood)]
