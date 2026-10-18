"""Domain features for heart_c, written from the schema and description only."""
import numpy as np
import pandas as pd

CP = "cp"


def _f(df, c):
    return pd.to_numeric(df[c].astype(object), errors="coerce").astype(float)


def _yes(s):
    return s.astype(object).map({"yes": 1.0, "no": 0.0, "t": 1.0, "f": 0.0}).astype(float)


def hr_reserve(df):
    # max heart rate vs age-predicted maximum: failing to reach it suggests chronotropic incompetence/ischemia
    # OUTSIDE KNOWLEDGE: age-predicted max heart rate formula 220 - age
    pred = 220 - _f(df, "age")
    return pd.DataFrame({"hr_pct_pred": _f(df, "thalach") / pred.replace(0, np.nan),
                         "hr_deficit": pred - _f(df, "thalach")})


def st_severity(df):
    # ST depression weighted by ST slope (flat/downsloping more ominous than upsloping)
    sl = df["slope"].astype(object).map({"up": 1.0, "flat": 2.0, "down": 3.0}).astype(float)
    return pd.DataFrame({"slope_ord": sl, "oldpeak_x_slope": _f(df, "oldpeak") * sl.fillna(2.0)})


def chest_pain(df):
    # asymptomatic chest pain is paradoxically the highest-risk category in this cohort; typical angina ranks next
    c = df[CP].astype(object)
    return pd.DataFrame({"cp_asympt": (c == "asympt").astype(float), "cp_typical": (c == "typ_angina").astype(float)})


def thal_defect(df):
    # thallium perfusion: reversible defect = inducible ischemia, fixed = scar
    t = df["thal"].astype(object)
    return pd.DataFrame({"thal_reversible": (t == "reversable_defect").astype(float),
                         "thal_any_defect": t.isin(["reversable_defect", "fixed_defect"]).astype(float)})


def ischemia_score(df):
    # count of exercise-test ischemia signals: angina, ST depression > 1, non-upsloping ST, vessels, thal defect, asympt cp
    s = (_yes(df["exang"]).fillna(0) + (_f(df, "oldpeak") > 1).astype(float)
         + df["slope"].astype(object).isin(["flat", "down"]).astype(float) + (_f(df, "ca") > 0).astype(float)
         + df["thal"].astype(object).isin(["reversable_defect", "fixed_defect"]).astype(float)
         + (df[CP].astype(object) == "asympt").astype(float))
    return s


def risk_factors(df):
    # classic modifiable risk factors: hypertension, hypercholesterolaemia, diabetes (fbs), male sex, age
    # OUTSIDE KNOWLEDGE: clinical cut-offs BP >= 140 mmHg, cholesterol >= 240 mg/dl
    s = ((_f(df, "trestbps") >= 140).astype(float) + (_f(df, "chol") >= 240).astype(float)
         + _yes(df["fbs"]).fillna(0) + (df["sex"].astype(object) == "male").astype(float) + (_f(df, "age") >= 55).astype(float))
    return pd.DataFrame({"risk_count": s, "age_x_male": _f(df, "age") * (df["sex"].astype(object) == "male")})


def rate_pressure(df):
    # double product (HR x systolic BP) approximates myocardial oxygen demand
    return _f(df, "thalach") * _f(df, "trestbps") / 1000.0


PROPOSALS = [("hr_reserve", hr_reserve), ("st_severity", st_severity), ("chest_pain", chest_pain),
             ("thal_defect", thal_defect), ("ischemia_score", ischemia_score), ("risk_factors", risk_factors),
             ("rate_pressure", rate_pressure)]
