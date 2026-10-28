"""Domain features for data, written from the schema and description only."""
import numpy as np
import pandas as pd

# OUTSIDE KNOWLEDGE: 1030 rows with these ranges match the concrete compressive-strength mix table:
# f_1 cement, f_2 blast-furnace slag, f_3 fly ash, f_4 water, f_5 superplasticizer, f_6 coarse aggregate,
# f_7 fine aggregate (kg/m3), f_8 age (days). Target is presumably compressive strength.


def water_cement(df):
    # Abrams' law: strength falls with water/cement ratio
    # OUTSIDE KNOWLEDGE: Abrams' law
    wc = df["f_4"] / df["f_1"].replace(0, np.nan)
    return pd.DataFrame({"w_c": wc, "c_w": df["f_1"] / df["f_4"].replace(0, np.nan)})


def water_binder(df):
    # supplementary cementitious materials (slag, fly ash) also bind; water/binder generalises w/c
    binder = df["f_1"] + df["f_2"] + df["f_3"]
    return pd.DataFrame({
        "binder": binder,
        "w_b": df["f_4"] / binder.replace(0, np.nan),
        "w_b_eff": df["f_4"] / (df["f_1"] + 0.9 * df["f_2"] + 0.4 * df["f_3"]).replace(0, np.nan),
    })


def binder_composition(df):
    # share of cement vs slag vs fly ash in the binder (replacement levels slow early strength)
    binder = (df["f_1"] + df["f_2"] + df["f_3"]).replace(0, np.nan)
    return pd.DataFrame({
        "cement_share": df["f_1"] / binder,
        "slag_share": df["f_2"] / binder,
        "ash_share": df["f_3"] / binder,
    })


def age_maturity(df):
    # strength gain is roughly logarithmic / hyperbolic in age
    # OUTSIDE KNOWLEDGE: ACI 209 strength-development curve t/(4+0.85t)
    t = df["f_8"].clip(lower=0)
    return pd.DataFrame({
        "log_age": np.log1p(t),
        "aci_maturity": t / (4.0 + 0.85 * t),
    })


def abrams_x_age(df):
    # interaction: binder/water quality times maturity (strength ~ f(w/b) * g(age))
    binder = df["f_1"] + df["f_2"] + df["f_3"]
    bw = binder / df["f_4"].replace(0, np.nan)
    t = df["f_8"].clip(lower=0)
    return pd.DataFrame({
        "bw_x_logage": bw * np.log1p(t),
        "cw_x_maturity": df["f_1"] / df["f_4"].replace(0, np.nan) * t / (4.0 + 0.85 * t),
    })


def aggregates_and_admixture(df):
    # paste vs aggregate proportions and superplasticizer dosage per binder
    binder = (df["f_1"] + df["f_2"] + df["f_3"]).replace(0, np.nan)
    agg = df["f_6"] + df["f_7"]
    total = df[["f_1", "f_2", "f_3", "f_4", "f_5", "f_6", "f_7"]].sum(axis=1).replace(0, np.nan)
    return pd.DataFrame({
        "sp_binder": df["f_5"] / binder,
        "agg_binder": agg / binder,
        "fine_share": df["f_7"] / agg.replace(0, np.nan),
        "paste_frac": (binder + df["f_4"]) / total,
        "has_sp": (df["f_5"] > 0).astype(float),
    })


PROPOSALS = [
    ("water_cement", water_cement),
    ("water_binder", water_binder),
    ("binder_composition", binder_composition),
    ("age_maturity", age_maturity),
    ("abrams_x_age", abrams_x_age),
    ("aggregates_and_admixture", aggregates_and_admixture),
]
