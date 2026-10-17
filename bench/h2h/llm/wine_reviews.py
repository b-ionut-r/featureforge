"""Domain features for wine_reviews, written from the schema and description only."""
import numpy as np
import pandas as pd


def _k(df, c):
    return df[c].astype(str).where(df[c].notna(), "NA")


def _rel(df, key):
    p = np.log(pd.to_numeric(df["price"], errors="coerce"))
    return p - p.groupby(key).transform("mean")


def log_price(df):
    # price is the strongest predictor of points and its effect is roughly logarithmic
    return np.log(pd.to_numeric(df["price"], errors="coerce"))


def price_rel_variety(df):
    # price relative to the same grape variety: an expensive Pinot is a premium bottling
    return _rel(df, _k(df, "variety"))


def price_rel_region(df):
    # price relative to the same appellation (region_1, falling back to province)
    key = _k(df, "region_1").where(df["region_1"].notna(), "P" + _k(df, "province"))
    return _rel(df, key)


def winery_stats(df):
    # winery size (number of reviewed wines) and its average price level: established premium producers score higher
    w = _k(df, "winery")
    p = np.log(pd.to_numeric(df["price"], errors="coerce"))
    return pd.DataFrame({"winery_n": w.groupby(w).transform("size").astype(float),
                         "winery_mean_logp": p.groupby(w).transform("mean"),
                         "price_rel_winery": p - p.groupby(w).transform("mean")})


def price_rel_taster(df):
    # each taster reviews a different price segment; price relative to the taster's usual wines
    return _rel(df, _k(df, "taster_name"))


def price_rel_country_variety(df):
    # price relative to country x variety peers
    return _rel(df, _k(df, "country") + "_" + _k(df, "variety"))


def counts(df):
    # popularity of variety and region in the review corpus
    v = _k(df, "variety")
    r = _k(df, "region_1")
    return pd.DataFrame({"variety_n": v.groupby(v).transform("size").astype(float),
                         "region_n": r.groupby(r).transform("size").astype(float).where(df["region_1"].notna())})


def price_missing(df):
    # missing price often means a non-retail / sample bottle
    return df["price"].isna().astype(float)


PROPOSALS = [("log_price", log_price), ("price_rel_variety", price_rel_variety), ("winery_stats", winery_stats),
             ("price_rel_region", price_rel_region), ("price_rel_taster", price_rel_taster),
             ("price_rel_country_variety", price_rel_country_variety), ("counts", counts), ("price_missing", price_missing)]
