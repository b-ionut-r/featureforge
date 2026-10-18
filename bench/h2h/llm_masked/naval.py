"""Domain features for data, written from the schema and description only."""
import numpy as np
import pandas as pd

# OUTSIDE KNOWLEDGE: the layout (9 lever positions f_1, speeds 3..27 kn f_2, 11934 rows, two identical torque
# columns) matches the naval-propulsion gas-turbine condition-monitoring simulation:
# f_3 GT shaft torque, f_4 GT rpm, f_5 gas-generator rpm, f_6/f_7 propeller torques, f_8 HP-turbine exit temp,
# f_9 compressor outlet temp, f_10 HP-turbine exit pressure, f_11 compressor outlet pressure, f_12 exhaust
# pressure, f_13 turbine injection control, f_14 fuel flow. The target is a degradation (decay) coefficient,
# which shows up as deviations of sensors from the healthy value at the same operating point.

_SENS = ["f_3", "f_4", "f_5", "f_6", "f_8", "f_9", "f_10", "f_11", "f_12", "f_13", "f_14"]


def _op(df):
    return df["f_2"].round(0)


def op_point_relative(df):
    # each sensor relative to the median at the same operating point (speed): decay = off-nominal readings
    g = _op(df)
    out = {}
    for c in _SENS:
        med = df.groupby(g)[c].transform("median")
        out[c + "_rel"] = df[c] / med.replace(0, np.nan) - 1.0
    return pd.DataFrame(out)


def op_point_zscore(df):
    # standardized deviation within operating point (scale-free across speeds)
    g = _op(df)
    out = {}
    for c in _SENS:
        grp = df.groupby(g)[c]
        out[c + "_z"] = (df[c] - grp.transform("mean")) / grp.transform("std").replace(0, np.nan)
    return pd.DataFrame(out)


def compressor_efficiency(df):
    # isentropic compressor efficiency from pressure ratio and outlet temperature (inlet ~288 K, ~1 bar)
    # OUTSIDE KNOWLEDGE: eta_c = T1*((P2/P1)^((g-1)/g) - 1) / (T2 - T1), gamma=1.4; T1=288.15 K, P1=0.998 bar
    T1, P1 = 288.15, 0.998
    pr = (df["f_11"] / P1).clip(lower=1e-6)
    eta = T1 * (pr ** (0.4 / 1.4) - 1.0) / (df["f_9"] - T1).replace(0, np.nan)
    g = _op(df)
    eta_rel = eta - eta.groupby(g).transform("median")
    return pd.DataFrame({"eta_c": eta, "eta_c_rel": eta_rel, "T2_over_T1": df["f_9"] / T1})


def turbine_efficiency(df):
    # HP-turbine expansion: temperature vs pressure drop across turbine; work per fuel
    # OUTSIDE KNOWLEDGE: Brayton-cycle relations, power = torque * angular speed
    pr_t = df["f_11"] / df["f_10"].replace(0, np.nan)
    power = df["f_3"] * df["f_4"] * (2 * np.pi / 60.0)
    sfc = df["f_14"] / power.replace(0, np.nan)
    g = _op(df)
    return pd.DataFrame({
        "turb_pr": pr_t,
        "T48_over_T2": df["f_8"] / df["f_9"].replace(0, np.nan),
        "sfc": sfc,
        "sfc_rel": sfc / sfc.groupby(g).transform("median").replace(0, np.nan) - 1.0,
        "fuel_per_tic": df["f_14"] / df["f_13"].replace(0, np.nan),
    })


def ratios_to_speed(df):
    # basic load normalisation: fuel, torque, temperatures per knot / per gas-generator rpm
    v = df["f_2"].replace(0, np.nan)
    return pd.DataFrame({
        "fuel_per_ggrpm": df["f_14"] / df["f_5"].replace(0, np.nan),
        "torque_per_rpm": df["f_3"] / df["f_4"].replace(0, np.nan),
        "ggrpm_per_rpm": df["f_5"] / df["f_4"].replace(0, np.nan),
        "p2_per_speed": df["f_11"] / v,
        "prop_over_gt_torque": df["f_6"] / df["f_3"].replace(0, np.nan),
    })


def ratio_deviations(df):
    # same physics ratios but relative to their operating-point median (isolates degradation)
    r = pd.concat([ratios_to_speed(df), compressor_efficiency(df)[["T2_over_T1"]]], axis=1)
    g = _op(df)
    out = {}
    for c in r.columns:
        out[c + "_rel"] = r[c] / r[c].groupby(g).transform("median").replace(0, np.nan) - 1.0
    return pd.DataFrame(out, index=df.index)


PROPOSALS = [
    ("op_point_relative", op_point_relative),
    ("op_point_zscore", op_point_zscore),
    ("compressor_efficiency", compressor_efficiency),
    ("turbine_efficiency", turbine_efficiency),
    ("ratios_to_speed", ratios_to_speed),
    ("ratio_deviations", ratio_deviations),
]
