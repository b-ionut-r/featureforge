"""Domain features for sf_crime, written from the schema and description only."""
import numpy as np
import pandas as pd


def _addr(df):
    return df["Address"].fillna("").astype(str).str.upper().str.strip()


def address_count(df):
    # incident volume at the address: hotspots (e.g. stores -> theft) differ in violent share from rare spots
    a = _addr(df)
    return a.groupby(a).transform("size").astype(float)


def is_intersection(df):
    # street-corner reports ("A / B") are more often street crime (assault, robbery) than block addresses
    return _addr(df).str.contains("/", regex=False).astype(float)


def block_number(df):
    # house-block number: low numbers sit near downtown/main corridors
    return pd.to_numeric(_addr(df).str.extract(r"^(\d+)\s+BLOCK", expand=False), errors="coerce")


def street_type(df):
    # street suffix (ST, AV, BL, WY...) of the first street: residential avenues vs commercial streets
    first = _addr(df).str.replace(r"^\d+\s+BLOCK OF\s+", "", regex=True).str.split("/").str[0].str.strip()
    suf = first.str.extract(r"\s([A-Z]{2})$", expand=False).fillna("NA")
    codes = {"ST": 0, "AV": 1, "BL": 2, "WY": 3, "DR": 4, "RD": 5, "TR": 6, "CT": 7, "PL": 8, "HY": 9, "LN": 10, "AL": 11}
    return suf.map(codes).fillna(12).astype(float)


def street_count(df):
    # incident volume of the main street (Market, Mission ... ) across all rows
    first = _addr(df).str.replace(r"^\d+\s+BLOCK OF\s+", "", regex=True).str.split("/").str[0].str.strip()
    return first.groupby(first).transform("size").astype(float)


def address_district_share(df):
    # how concentrated the district's incidents are at this address
    a = _addr(df)
    n_a = a.groupby(a).transform("size")
    n_d = df["PdDistrict"].astype(str).groupby(df["PdDistrict"].astype(str)).transform("size")
    return n_a / n_d


def night_weekend(df):
    # violent crime peaks late night and on weekend nights (bars), property crime by day
    h = pd.to_numeric(df["Hour"], errors="coerce")
    d = pd.to_numeric(df["DayOfWeek"], errors="coerce")
    night = ((h >= 21) | (h <= 3)).astype(float)
    return pd.DataFrame({"night": night, "hour_sin": np.sin(2 * np.pi * h / 24), "hour_cos": np.cos(2 * np.pi * h / 24),
                         "night_x_weekend": night * d.isin([1, 6, 7]).astype(float)})


def address_year_span(df):
    # number of distinct years the address appears in: chronic hotspot vs one-off
    a = _addr(df)
    return df["Year"].astype(str).groupby(a).transform("nunique").astype(float)


PROPOSALS = [("address_count", address_count), ("is_intersection", is_intersection), ("street_count", street_count),
             ("address_district_share", address_district_share), ("night_weekend", night_weekend),
             ("block_number", block_number), ("street_type", street_type), ("address_year_span", address_year_span)]
