"""Domain features for hepatitis, written from the schema and description only."""
import numpy as np
import pandas as pd


def _f(df, c):
    return pd.to_numeric(df[c].astype(object), errors="coerce").astype(float)


def _yn(df, c):
    s = df[c].astype(object).astype(str).str.strip().str.lower()
    return pd.Series(np.where(s == "yes", 1.0, np.where(s == "no", 0.0, np.nan)), index=df.index)


def decompensation_signs(df):
    # signs of portal hypertension / decompensated cirrhosis: ascites, varices, spiders, palpable spleen
    cols = ["ASCITES", "VARICES", "SPIDERS", "SPLEEN_PALPABLE"]
    m = pd.DataFrame({c: _yn(df, c) for c in cols})
    return pd.DataFrame({"n_decomp": m.sum(axis=1, min_count=1), "ascites_or_varices": m[["ASCITES", "VARICES"]].max(axis=1)})


def symptom_count(df):
    # constitutional symptom burden: fatigue, malaise, anorexia
    return pd.DataFrame({c: _yn(df, c) for c in ["FATIGUE", "MALAISE", "ANOREXIA"]}).sum(axis=1, min_count=1)


def liver_synthetic(df):
    # synthetic function failure: low albumin, prolonged prothrombin (low PROTIME % activity), high bilirubin
    alb, bil, pt = _f(df, "ALBUMIN"), _f(df, "BILIRUBIN"), _f(df, "PROTIME")
    return pd.DataFrame({"bili_over_alb": bil / alb.replace(0, np.nan), "alb_x_pt": alb * pt / 100.0,
                         "log_bili": np.log(bil.clip(lower=0.1))})


def albi_like(df):
    # ALBI-style grade: combines log bilirubin and albumin into one liver reserve score
    # OUTSIDE KNOWLEDGE: ALBI = 0.66*log10(bili umol/L) - 0.085*alb g/L; bili mg/dL*17.1, alb g/dL*10
    bil, alb = _f(df, "BILIRUBIN"), _f(df, "ALBUMIN")
    return 0.66 * np.log10((bil * 17.1).clip(lower=1)) - 0.085 * alb * 10


def enzyme_ratio(df):
    # hepatocellular (SGOT) vs cholestatic (alk phosphate) injury pattern
    return np.log1p(_f(df, "SGOT").clip(lower=0)) - np.log1p(_f(df, "ALK_PHOSPHATE").clip(lower=0))


def lab_abnormal_count(df):
    # count of abnormal labs using common reference limits; missing labs counted separately (sicker/limited workup)
    # OUTSIDE KNOWLEDGE: approximate normal ranges (bili>1.2, alk>120, SGOT>40, albumin<3.5, protime<60)
    bil, alk, sg, alb, pt = (_f(df, c) for c in ["BILIRUBIN", "ALK_PHOSPHATE", "SGOT", "ALBUMIN", "PROTIME"])
    ab = (bil > 1.2).astype(float) + (alk > 120).astype(float) + (sg > 40).astype(float) + (alb < 3.5).astype(float) + (pt < 60).astype(float)
    miss = pd.DataFrame({"a": bil, "b": alk, "c": sg, "d": alb, "e": pt}).isna().sum(axis=1).astype(float)
    return pd.DataFrame({"n_abnormal": ab, "n_lab_missing": miss})


def age_x_severity(df):
    # older patients with cirrhosis signs (firm liver + histology done) fare worse
    return pd.DataFrame({"age_x_alb": _f(df, "AGE") / _f(df, "ALBUMIN").replace(0, np.nan),
                         "firm_x_histology": _yn(df, "LIVER_FIRM") * _yn(df, "HISTOLOGY")})


def treatment_flags(df):
    # treatment and sex mapped to numbers (steroid/antiviral use reflects disease course)
    sex = df["SEX"].astype(object).astype(str).str.lower()
    return pd.DataFrame({"steroid": _yn(df, "STEROID"), "antivirals": _yn(df, "ANTIVIRALS"),
                         "male": np.where(sex == "male", 1.0, np.where(sex == "female", 0.0, np.nan))}, index=df.index)


PROPOSALS = [("decompensation_signs", decompensation_signs), ("symptom_count", symptom_count),
             ("liver_synthetic", liver_synthetic), ("albi_like", albi_like), ("enzyme_ratio", enzyme_ratio),
             ("lab_abnormal_count", lab_abnormal_count), ("age_x_severity", age_x_severity),
             ("treatment_flags", treatment_flags)]
