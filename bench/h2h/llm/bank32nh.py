"""Domain features for bank32nh, written from the schema and description only.

Column meanings are inferred from names: a{i}cx/cy/sx/sy/rho/pop = residential area i centre, spread, correlation,
population; b{j}x/y/call/eff = bank j location, number of tellers(?) and teller effectiveness; mxql = max queue length.
"""
import numpy as np
import pandas as pd


def _dist(df, i, j):
    return np.sqrt((df[f"a{i}cx"] - df[f"b{j}x"]) ** 2 + (df[f"a{i}cy"] - df[f"b{j}y"]) ** 2)


def area_bank_dists(df):
    # customers choose banks by distance: all 9 area-to-bank distances
    return pd.DataFrame({f"d_a{i}_b{j}": _dist(df, i, j) for i in (1, 2, 3) for j in (1, 2, 3)})


def bank_demand(df):
    # gravity-model demand on each bank: population shared by inverse distance
    out = {}
    for j in (1, 2, 3):
        out[f"b{j}_demand"] = 0.0
    for i in (1, 2, 3):
        w = {j: 1 / (_dist(df, i, j) + 0.1) for j in (1, 2, 3)}
        tot = w[1] + w[2] + w[3]
        for j in (1, 2, 3):
            out[f"b{j}_demand"] = out[f"b{j}_demand"] + df[f"a{i}pop"] * w[j] / tot
    return pd.DataFrame(out, index=df.index)


def bank_capacity(df):
    # service capacity = tellers x effectiveness x max queue length
    return pd.DataFrame({f"b{j}_cap": df[f"b{j}call"].astype(float) * df[f"b{j}eff"] * df["mxql"] for j in (1, 2, 3)})


def load_ratio(df):
    # demand over capacity per bank and the worst one: rejections happen when queues are full
    d, c = bank_demand(df), bank_capacity(df)
    r = pd.DataFrame({f"b{j}_load": d[f"b{j}_demand"] / c[f"b{j}_cap"].replace(0, np.nan) for j in (1, 2, 3)})
    r["max_load"] = r.max(axis=1)
    r["total_load"] = d.sum(axis=1) / c.sum(axis=1).replace(0, np.nan)
    return r


def total_pop_capacity(df):
    # system-wide population vs teller capacity, scaled by temp (arrival-rate-like parameter?)
    pop = df["a1pop"] + df["a2pop"] + df["a3pop"]
    cap = sum(df[f"b{j}call"].astype(float) * df[f"b{j}eff"] for j in (1, 2, 3))
    return pd.DataFrame({"total_pop": pop, "pop_per_cap": pop / cap.replace(0, np.nan),
                         "temp_pop_per_cap": df["temp"] * pop / cap.replace(0, np.nan)})


def area_spread(df):
    # area spread (sx*sy) and population density
    return pd.DataFrame({f"a{i}_density": df[f"a{i}pop"] / (df[f"a{i}sx"] * df[f"a{i}sy"] * np.sqrt((1 - df[f"a{i}rho"] ** 2).clip(lower=1e-3))).replace(0, np.nan)
                         for i in (1, 2, 3)})


PROPOSALS = [("area_bank_dists", area_bank_dists), ("bank_demand", bank_demand), ("bank_capacity", bank_capacity),
             ("load_ratio", load_ratio), ("total_pop_capacity", total_pop_capacity), ("area_spread", area_spread)]
