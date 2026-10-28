"""Domain features for data, written from the schema and description only."""
import numpy as np
import pandas as pd

# Columns look like a dress catalogue: f_1 style, f_2 price level, f_3 rating, f_4 size, f_5 season,
# f_6 neckline, f_7 sleeve length, f_8 waistline, f_9 material, f_10 fabric, f_11 decoration, f_12 pattern.
# Category spellings are dirty (case, typos), so everything is normalised first.


def _norm(s):
    return s.astype(str).str.strip().str.lower()


def price_level(df):
    # price tier as an ordered number (low < average/medium < high < very-high)
    m = {"low": 1, "average": 2, "medium": 2.5, "high": 3, "very-high": 4, "very high": 4}
    return _norm(df["f_2"]).map(m).astype(float).rename("price_ord")


def size_level(df):
    # size as ordered number; 'free' (one-size) placed in the middle
    m = {"s": 1, "small": 1, "m": 2, "l": 3, "xl": 4, "free": 2.5}
    return _norm(df["f_4"]).map(m).astype(float).rename("size_ord")


def rating_features(df):
    # rating 0.0 almost surely means 'not rated yet' rather than terrible; separate the two signals
    r = pd.to_numeric(df["f_3"], errors="coerce")
    return pd.DataFrame({
        "unrated": (r.fillna(0) <= 0).astype(float),
        "rating_if_rated": r.where(r > 0),
    })


def season_clean(df):
    # season with typos merged (Automn/Autumn, spring/Spring), one column per season
    s = _norm(df["f_5"]).replace({"automn": "autumn"})
    out = {}
    for k in ["summer", "spring", "autumn", "winter"]:
        out["season_" + k] = (s == k).astype(float)
    return pd.DataFrame(out)


def sleeve_length(df):
    # sleeve coverage as ordered amount of arm covered
    s = _norm(df["f_7"])
    m = {"sleevless": 0, "sleeveless": 0, "sleeevless": 0, "sleveless": 0, "capsleeves": 0.5, "cap-sleeves": 0.5,
         "butterfly": 0.5, "petal": 0.5, "short": 1, "halfsleeve": 2, "half": 2, "threequarter": 3,
         "threequater": 3, "thressqatar": 3, "three-quarter": 3, "turndowncollor": np.nan, "full": 4}
    return s.map(m).astype(float).rename("sleeve_ord")


def detail_completeness(df):
    # how fully specified the listing is: count of missing optional attributes (waist, material, fabric, decoration, pattern)
    cols = ["f_8", "f_9", "f_10", "f_11", "f_12"]
    miss = pd.DataFrame({c: df[c].isna() | _norm(df[c]).isin(["nan", "none", "null", ""]) for c in cols})
    return pd.DataFrame({
        "n_missing_attrs": miss.sum(axis=1).astype(float),
        "has_decoration": (~miss["f_11"]).astype(float),
    })


def style_frequency(df):
    # popularity of the (normalised) style within the catalogue; label-free frequency encoding
    s = _norm(df["f_1"]).replace({"sexy": "sexy"})
    return s.map(s.value_counts(normalize=True)).astype(float).rename("style_freq")


def price_x_rating(df):
    # value-for-money proxy: rating relative to price tier
    p = price_level(df)
    r = pd.to_numeric(df["f_3"], errors="coerce")
    r = r.where(r > 0)
    return (r / p.replace(0, np.nan)).rename("rating_per_price")


PROPOSALS = [
    ("price_level", price_level),
    ("size_level", size_level),
    ("rating_features", rating_features),
    ("season_clean", season_clean),
    ("sleeve_length", sleeve_length),
    ("detail_completeness", detail_completeness),
    ("style_frequency", style_frequency),
    ("rating_per_price", price_x_rating),
]
