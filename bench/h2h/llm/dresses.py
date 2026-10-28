"""Domain features for dresses, written from the schema and description only."""
import numpy as np
import pandas as pd


def _clean(s):
    return s.astype(str).str.strip().str.lower().replace({"none": np.nan, "nan": np.nan, "null": np.nan, "": np.nan})


# Price tier as an ordered number (V3), normalizing case/typos.
def price_level(df):
    m = {"low": 1.0, "average": 2.0, "medium": 3.0, "high": 4.0, "very-high": 5.0, "very high": 5.0}
    return _clean(df["V3"]).map(m).astype(float)


# Rating (V4): 0 appears to mean "not yet rated", so split into rated flag and rating-when-rated.
def rating_parts(df):
    r = pd.to_numeric(df["V4"], errors="coerce")
    rated = (r > 0).astype(float)
    return pd.DataFrame({"is_rated": rated, "rating_if_rated": r.where(r > 0)}, index=df.index)


# Size (V5) as ordered numbers; "free" size is a separate one-size-fits-all flag.
def size_level(df):
    s = _clean(df["V5"])
    m = {"s": 1.0, "small": 1.0, "m": 2.0, "l": 3.0, "xl": 4.0}
    return pd.DataFrame({"size_ord": s.map(m).astype(float), "size_free": (s == "free").astype(float)}, index=df.index)


# Season (V6) on a warm-cold scale: summer/spring dresses sell differently from autumn/winter ones.
def season_warmth(df):
    s = _clean(df["V6"])
    m = {"summer": 3.0, "spring": 2.0, "automn": 1.0, "autumn": 1.0, "winter": 0.0}
    return s.map(m).astype(float)


# Sleeve coverage (V8) as ordered amount of fabric on the arm.
def sleeve_coverage(df):
    s = _clean(df["V8"])
    m = {"sleevless": 0.0, "sleeveless": 0.0, "sleveless": 0.0, "cap-sleeves": 1.0, "capsleeves": 1.0,
         "short": 1.0, "butterfly": 1.0, "petal": 1.0, "halfsleeve": 2.0, "half": 2.0, "halfsleeves": 2.0,
         "threequarter": 3.0, "threequater": 3.0, "thressqatar": 3.0, "full": 4.0, "turndowncollor": np.nan}
    return s.map(m).astype(float)


# How fully described a listing is: number of filled attributes among the sparse descriptive fields.
def n_described(df):
    cols = ["V7", "V9", "V10", "V11", "V12", "V13"]
    return df[cols].apply(lambda c: _clean(c).notna().astype(float)).sum(axis=1)


# Popularity of each attribute value across the catalogue (label-free frequency encoding): mainstream vs niche designs.
def value_frequencies(df):
    out = {}
    for c in ["V2", "V7", "V8", "V10", "V11", "V12", "V13"]:
        s = _clean(df[c]).fillna("__na__")
        out["freq_" + c] = s.map(s.value_counts(normalize=True)).astype(float)
    return pd.DataFrame(out, index=df.index)


# Style group (V2): occasion styles (party, sexy) vs everyday ones (casual, brief, work).
def style_flags(df):
    s = _clean(df["V2"])
    return pd.DataFrame({
        "style_occasion": s.isin(["party", "sexy", "flare"]).astype(float),
        "style_everyday": s.isin(["casual", "brief", "work", "ol"]).astype(float),
    }, index=df.index)


PROPOSALS = [
    ("price_level", price_level),
    ("rating_parts", rating_parts),
    ("size_level", size_level),
    ("season_warmth", season_warmth),
    ("sleeve_coverage", sleeve_coverage),
    ("n_described", n_described),
    ("value_frequencies", value_frequencies),
    ("style_flags", style_flags),
]
