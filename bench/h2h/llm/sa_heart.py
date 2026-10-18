"""Domain features for sa_heart, written from the schema and description only."""
import numpy as np
import pandas as pd

# Columns decoded from the description's attribute list (in order): V1 sbp, V2 tobacco (kg cumulative), V3 ldl,
# V4 adiposity, V5 famhist (2 levels), V6 typea, V7 obesity (BMI-like), V8 alcohol, V9 age.


def _fam(df):
    return (df["V5"].astype(object).astype(str) == "1").astype(float)


def tobacco_rate(df):
    # cumulative tobacco relative to age: intensity of smoking (kg per year of life), plus any-smoker flag
    return pd.DataFrame({"tob_per_age": df["V2"] / df["V9"].replace(0, np.nan), "log_tobacco": np.log1p(df["V2"]),
                         "smoker": (df["V2"] > 0).astype(float)})


def lipid_age(df):
    # LDL exposure accumulates with age (LDL-years); atherosclerosis builds over time
    return pd.DataFrame({"ldl_x_age": df["V3"] * df["V9"] / 10.0, "ldl_high": (df["V3"] >= 5.0).astype(float)})


def bp_risk(df):
    # systolic hypertension and pulse-pressure-like age interaction
    # OUTSIDE KNOWLEDGE: hypertension threshold 140 mmHg
    return pd.DataFrame({"htn": (df["V1"] >= 140).astype(float), "sbp_x_age": df["V1"] * df["V9"] / 100.0})


def body_fat(df):
    # adiposity vs BMI-like obesity: fat share per unit BMI (central/visceral adiposity) 
    return pd.DataFrame({"adip_over_obes": df["V4"] / df["V7"].replace(0, np.nan), "adip_minus_obes": df["V4"] - df["V7"]})


def risk_count(df):
    # count of classic CHD risk factors: hypertension, smoking, high LDL, family history, older age, obesity
    s = ((df["V1"] >= 140).astype(float) + (df["V2"] > 0).astype(float) + (df["V3"] >= 5.0).astype(float)
         + (df["V9"] >= 50).astype(float) + (df["V7"] >= 30).astype(float))
    return pd.DataFrame({"risk_count": s, "fam1": _fam(df), "fam1_x_age": _fam(df) * df["V9"]})


def framingham_like(df):
    # log-additive score in the spirit of Framingham: age, sbp, ldl, smoking dominate
    # OUTSIDE KNOWLEDGE: Framingham-style log-linear risk structure (weights here are rough, not fitted)
    return (0.05 * df["V9"] + 0.01 * df["V1"] + 0.2 * df["V3"] + 0.3 * np.log1p(df["V2"])
            + 0.02 * df["V6"])


def alcohol(df):
    # alcohol: log scale and heavy-drinker flag (U-shaped cardio association)
    return pd.DataFrame({"log_alcohol": np.log1p(df["V8"]), "heavy_alcohol": (df["V8"] > 30).astype(float)})


PROPOSALS = [("tobacco_rate", tobacco_rate), ("lipid_age", lipid_age), ("bp_risk", bp_risk), ("body_fat", body_fat),
             ("risk_count", risk_count), ("framingham_like", framingham_like), ("alcohol", alcohol)]
