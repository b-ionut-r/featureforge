"""Domain features for coil2000, written from the schema and description only."""
import numpy as np
import pandas as pd

# CoIL 2000 caravan-insurance data: M* = postcode demographics (0-9 bands), P* = contribution level
# per policy type, A* = number of policies per type.
P_COLS = ["PWAPART", "PWABEDR", "PWALAND", "PPERSAUT", "PBESAUT", "PMOTSCO", "PVRAAUT", "PAANHANG",
          "PTRACTOR", "PWERKT", "PBROM", "PLEVEN", "PPERSONG", "PGEZONG", "PWAOREG", "PBRAND",
          "PZEILPL", "PPLEZIER", "PFIETS", "PINBOED", "PBYSTAND"]
A_COLS = ["A" + c[1:] for c in P_COLS]


def _cols(df, cols):
    return df[[c for c in cols if c in df.columns]].astype(float)


def total_contribution(df):
    # customers who already spend a lot on insurance with the company are the ones who buy add-on policies
    return _cols(df, P_COLS).sum(axis=1)


def n_policy_types(df):
    # breadth of the existing relationship: number of distinct product lines held
    return (_cols(df, P_COLS) > 0).sum(axis=1).astype(float)


def car_x_fire(df):
    # car policy plus fire policy marks an established private household customer (the caravan buyer profile)
    return df["PPERSAUT"].astype(float) * df["PBRAND"].astype(float)


def car_contrib_x_purchasing_power(df):
    # a high car-policy contribution only signals a caravan owner when the household can afford one
    return df["PPERSAUT"].astype(float) * df["MKOOPKLA"].astype(float)


def total_policies(df):
    # total count of policies held across all product lines
    return _cols(df, A_COLS).sum(axis=1)


def wealth_index(df):
    # postcode affluence: high-income + home owners + 1 car minus low-income/renters/no car
    return (df["MINK7512"] + df["MINK123M"] + df["MHKOOP"] + df["MAUT1"] + df["MKOOPKLA"]
            - df["MINKM30"] - df["MHHUUR"] - df["MAUT0"]).astype(float)


def mostype_count(df):
    # size of the customer subtype segment: rare segments behave differently
    return df.groupby("MOSTYPE")["MOSTYPE"].transform("size").astype(float)


def car_contrib_vs_subtype(df):
    # car-policy contribution relative to peers of the same customer subtype
    x = df["PPERSAUT"].astype(float)
    return x - x.groupby(df["MOSTYPE"]).transform("mean")


PROPOSALS = [
    ("car_x_fire", car_x_fire),
    ("car_contrib_x_purchasing_power", car_contrib_x_purchasing_power),
    ("total_contribution", total_contribution),
    ("n_policy_types", n_policy_types),
    ("total_policies", total_policies),
    ("wealth_index", wealth_index),
    ("car_contrib_vs_subtype", car_contrib_vs_subtype),
    ("mostype_count", mostype_count),
]
