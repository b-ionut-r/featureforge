"""Domain features for nomao, written from the schema and description only."""
import numpy as np
import pandas as pd

# Pairs of place records; each field (name, city, zip, street, website, ..., phone, fax, coords)
# is compared with several similarity measures (V1-V6 name, V9-V14 city, V17-V22 zip, V25-V30 street,
# V33-V38 website, V57-V62/V65-V70 geocoded address, V89-V91 phone, V93-V95 fax, V97-V99 street number,
# V101-V118 coordinate similarities) plus nominal including/equality flags (n,s,m coded 1/2/3).

_BLOCKS = {"name": range(1, 7), "city": range(9, 15), "zip": range(17, 23), "street": range(25, 31),
           "website": range(33, 39), "country": range(41, 47), "locality": range(49, 55),
           "in_addr": range(57, 63), "out_addr": range(65, 71), "postcode": range(73, 79),
           "phone": range(89, 92), "fax": range(93, 96), "street_no": range(97, 100)}


def _num(df, cols):
    cols = [f"V{i}" for i in cols if f"V{i}" in df.columns]
    return df[cols].apply(pd.to_numeric, errors="coerce")


def _numeric_cols(df):
    return [c for c in df.columns if df[c].dtype.kind == "f"]


def _nominal_cols(df):
    return [c for c in df.columns if df[c].dtype == object]


def field_best_similarity(df):
    # best similarity per field: a duplicate needs agreement on at least one strong identifier per field
    return pd.DataFrame({"best_" + k: _num(df, r).max(axis=1) for k, r in _BLOCKS.items()}, index=df.index)


def overall_agreement(df):
    # overall evidence of being the same place: mean/min similarity and counts of perfect and zero matches
    x = df[_numeric_cols(df)].apply(pd.to_numeric, errors="coerce")
    return pd.DataFrame({"sim_mean": x.mean(axis=1), "sim_min": x.min(axis=1),
                         "n_exact": (x >= 0.999).sum(axis=1).astype(float),
                         "n_zero": (x <= 1e-9).sum(axis=1).astype(float)}, index=df.index)


def nominal_flag_counts(df):
    # how many including/equality flags take each code across fields (e.g. how many fields are 'equal')
    s = df[_nominal_cols(df)].astype(object).fillna("na").astype(str)
    return pd.DataFrame({"n_code1": (s == "1").sum(axis=1).astype(float),
                         "n_code2": (s == "2").sum(axis=1).astype(float),
                         "n_code3": (s == "3").sum(axis=1).astype(float)}, index=df.index)


def name_and_location(df):
    # same name AND close coordinates is the classic duplicate; same name far away = chain store branch
    name = _num(df, range(1, 7)).max(axis=1)
    geo = _num(df, [117, 118]).mean(axis=1)
    street = _num(df, range(25, 31)).max(axis=1)
    phone = _num(df, range(89, 92)).max(axis=1)
    return pd.DataFrame({"name_x_geo": name * geo, "name_x_street": name * street,
                         "name_x_phone": name * phone, "name_minus_street": name - street}, index=df.index)


def name_measure_spread(df):
    # disagreement between name similarity measures (levenshtein vs trigram vs token): typos vs different names
    x = _num(df, range(1, 7))
    return pd.DataFrame({"name_sim_std": x.std(axis=1), "name_mean": x.mean(axis=1),
                         "name_lev_minus_tri": x["V3"] - x["V4"]}, index=df.index)


PROPOSALS = [
    ("name_and_location", name_and_location),
    ("overall_agreement", overall_agreement),
    ("field_best_similarity", field_best_similarity),
    ("nominal_flag_counts", nominal_flag_counts),
    ("name_measure_spread", name_measure_spread),
]
