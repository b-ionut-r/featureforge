"""Domain features for sulfur, written from the schema and description only.

Sulfur recovery unit: a1..a5 are (scaled) gas and air flows, y2 is the other process output.
Claus chemistry is governed by the air-to-acid-gas ratio, so features are flow ratios and balances.
"""
import numpy as np
import pandas as pd

# OUTSIDE KNOWLEDGE: in Fortuna et al.'s SRU data the inputs are, in order: MEA gas flow, MEA air flow,
# secondary (trim) air flow, SWS gas flow, SWS air flow; outputs are H2S and SO2 concentrations.
EPS = 1e-3


def air_gas_ratio_mea(df):
    # MEA line air-to-gas ratio controls H2S/SO2 stoichiometry
    return df["a2"] / (df["a1"] + EPS)


def air_gas_ratio_sws(df):
    # SWS line air-to-gas ratio
    return df["a5"] / (df["a4"] + EPS)


def total_air_over_gas(df):
    # overall air (incl. secondary air) per unit acid gas: excess air -> SO2, deficit -> H2S
    return (df["a2"] + df["a3"] + df["a5"]) / (df["a1"] + df["a4"] + EPS)


def air_gas_balance(df):
    # additive balance (robust to the min-max scaling) between air and gas flows
    return pd.DataFrame({"air_minus_gas": (df["a2"] + df["a3"] + df["a5"]) - (df["a1"] + df["a4"]),
                         "mea_air_minus_gas": df["a2"] - df["a1"], "sws_air_minus_gas": df["a5"] - df["a4"]})


def secondary_air_share(df):
    # trim air as share of total air: the controller's fine correction
    return df["a3"] / (df["a2"] + df["a3"] + df["a5"] + EPS)


def totals(df):
    # total gas load and total air flow (plant throughput)
    return pd.DataFrame({"total_gas": df["a1"] + df["a4"], "total_air": df["a2"] + df["a3"] + df["a5"]})


def so2_vs_air(df):
    # H2S and SO2 move oppositely with air ratio; y2 relative to air excess carries the residual state
    return pd.DataFrame({"y2_x_airratio": df["y2"] * total_air_over_gas(df),
                         "y2_minus_airbal": df["y2"] - ((df["a2"] + df["a3"] + df["a5"]) - (df["a1"] + df["a4"]))})


PROPOSALS = [("air_gas_ratio_mea", air_gas_ratio_mea), ("air_gas_ratio_sws", air_gas_ratio_sws),
             ("total_air_over_gas", total_air_over_gas), ("air_gas_balance", air_gas_balance),
             ("secondary_air_share", secondary_air_share), ("totals", totals), ("so2_vs_air", so2_vs_air)]
