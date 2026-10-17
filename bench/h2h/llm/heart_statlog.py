"""Domain features for heart_statlog, written from the schema and description only."""
import numpy as np
import pandas as pd


def _f(df, c):
    return pd.to_numeric(df[c].astype(object), errors="coerce").astype(float)


def hr_reserve(df):
    # max heart rate vs age-predicted maximum: failing to reach it suggests chronotropic incompetence/ischemia
    # OUTSIDE KNOWLEDGE: age-predicted max heart rate formula 220 - age
    pred = 220 - _f(df, "age")
    return pd.DataFrame({"hr_pct_pred": _f(df, "maximum_heart_rate_achieved") / pred.replace(0, np.nan),
                         "hr_deficit": pred - _f(df, "maximum_heart_rate_achieved")})


def st_severity(df):
    # ST depression weighted by slope (flat/downsloping ST is more ominous than upsloping)
    # OUTSIDE KNOWLEDGE: Cleveland coding slope 1=up, 2=flat, 3=down
    return _f(df, "oldpeak") * _f(df, "slope")


def asymptomatic_chest(df):
    # chest pain type 4 (asymptomatic) is paradoxically the highest-risk category in this cohort's coding
    # OUTSIDE KNOWLEDGE: chest pain codes 1=typical angina, 2=atypical, 3=non-anginal, 4=asymptomatic
    c = _f(df, "chest")
    return pd.DataFrame({"chest_asympt": (c == 4).astype(float), "chest_typical": (c == 1).astype(float)})


def thal_defect(df):
    # thal 6/7 = fixed/reversible perfusion defect vs 3 normal (from the description)
    t = _f(df, "thal")
    return pd.DataFrame({"thal_reversible": (t == 7).astype(float), "thal_any_defect": (t > 3).astype(float)})


def ischemia_score(df):
    # count of exercise-test ischemia signals: angina, ST depression > 1, non-upsloping ST, vessels, thal defect
    s = (_f(df, "exercise_induced_angina") + (_f(df, "oldpeak") > 1).astype(float)
         + (_f(df, "slope") >= 2).astype(float) + (_f(df, "number_of_major_vessels") > 0).astype(float)
         + (_f(df, "thal") > 3).astype(float))
    return s


def risk_factor_load(df):
    # classic risk-factor burden: age, male sex, high BP, high cholesterol, diabetes proxy
    return ((_f(df, "age") >= 55).astype(float) + _f(df, "sex") + (_f(df, "resting_blood_pressure") >= 140).astype(float)
            + (_f(df, "serum_cholestoral") >= 240).astype(float) + _f(df, "fasting_blood_sugar"))


def rate_pressure(df):
    # double product proxy (max HR x resting SBP) and cholesterol per year of age
    return pd.DataFrame({"rpp": _f(df, "maximum_heart_rate_achieved") * _f(df, "resting_blood_pressure") / 1000.0,
                         "chol_per_age": _f(df, "serum_cholestoral") / _f(df, "age").replace(0, np.nan)})


def vessels_x_age(df):
    # fluoroscopy-visible vessel disease interacting with age
    return _f(df, "number_of_major_vessels") * _f(df, "age") / 10.0


PROPOSALS = [("hr_reserve", hr_reserve), ("st_severity", st_severity), ("asymptomatic_chest", asymptomatic_chest),
             ("thal_defect", thal_defect), ("ischemia_score", ischemia_score), ("risk_factor_load", risk_factor_load),
             ("rate_pressure", rate_pressure), ("vessels_x_age", vessels_x_age)]
