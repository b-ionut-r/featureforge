"""Domain features for diabetes130, written from the schema and description only."""
import numpy as np
import pandas as pd

MEDS = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide",
        "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
        "troglitazone", "tolazamide", "examide", "citoglipton", "insulin", "glyburide.metformin",
        "glipizide.metformin", "glimepiride.pioglitazone", "metformin.rosiglitazone", "metformin.pioglitazone"]


def _meds(df):
    return df[[c for c in MEDS if c in df.columns]].astype(str)


def prior_utilization(df):
    # prior-year inpatient, emergency and outpatient visits: the best-known readmission predictor
    return pd.DataFrame({
        "total_visits": (df["number_inpatient"] + df["number_emergency"] + df["number_outpatient"]).astype(float),
        "acute_visits": (df["number_inpatient"] + df["number_emergency"]).astype(float),
    }, index=df.index)


def patient_encounter_count(df):
    # how many encounters this patient has in the data: frequent flyers get readmitted
    return df.groupby("patient_nbr")["patient_nbr"].transform("size").astype(float)


def patient_encounter_rank(df):
    # position of this encounter among the patient's encounters (encounter ids increase over time)
    return df.groupby("patient_nbr")["encounter_id"].rank(method="first").astype(float)


def icd9_group(df):
    # map ICD-9 codes of the three diagnoses to clinical chapters (circulatory, respiratory, diabetes, ...)
    def grp(s):
        s = s.astype(str).str.strip()
        v = pd.to_numeric(s, errors="coerce")
        out = pd.Series(8.0, index=s.index)  # other
        out[s.str.startswith(("V", "E"))] = 9.0
        out[((v >= 390) & (v < 460)) | ((v >= 785) & (v < 786))] = 0.0
        out[((v >= 460) & (v < 520)) | ((v >= 786) & (v < 787))] = 1.0
        out[((v >= 520) & (v < 580)) | ((v >= 787) & (v < 788))] = 2.0
        out[(v >= 250) & (v < 251)] = 3.0
        out[(v >= 800) & (v < 1000)] = 4.0
        out[(v >= 710) & (v < 740)] = 5.0
        out[((v >= 580) & (v < 630)) | ((v >= 788) & (v < 789))] = 6.0
        out[(v >= 140) & (v < 240)] = 7.0
        out[s.isin(["?", "nan", "None"])] = np.nan
        return out
    return pd.DataFrame({f"{c}_group": grp(df[c]) for c in ["diag_1", "diag_2", "diag_3"]}, index=df.index)


def med_changes(df):
    # number of diabetes drugs dosed up/down and number in use: unstable glycaemic control
    m = _meds(df)
    return pd.DataFrame({
        "n_med_changes": m.isin(["Up", "Down"]).sum(axis=1).astype(float),
        "n_meds_used": (m != "No").sum(axis=1).astype(float),
    }, index=df.index)


def age_mid(df):
    # midpoint of the age band in years
    lo = pd.to_numeric(df["age"].astype(str).str.extract(r"\[(\d+)")[0], errors="coerce")
    return (lo + 5).astype(float)


def intensity_per_day(df):
    # labs and medications per day in hospital: how acute the stay was
    t = df["time_in_hospital"].astype(float).replace(0, np.nan)
    return pd.DataFrame({"labs_per_day": df["num_lab_procedures"] / t,
                         "meds_per_day": df["num_medications"] / t}, index=df.index)


def patient_prior_inpatient_max(df):
    # the patient's highest prior inpatient count across their encounters in the data
    return df.groupby("patient_nbr")["number_inpatient"].transform("max").astype(float)


def discharge_risk(df):
    # discharged to another facility / home health (more fragile) vs home vs expired/hospice
    d = df["discharge_disposition_id"].astype(float)
    out = pd.Series(1.0, index=df.index)
    out[d == 1] = 0.0
    out[d.isin([3, 4, 5, 6, 22, 23, 24])] = 2.0
    out[d.isin([11, 13, 14, 19, 20, 21])] = -1.0
    return out


PROPOSALS = [
    ("prior_utilization", prior_utilization),
    ("patient_encounter_count", patient_encounter_count),
    ("patient_encounter_rank", patient_encounter_rank),
    ("patient_prior_inpatient_max", patient_prior_inpatient_max),
    ("discharge_risk", discharge_risk),
    ("icd9_group", icd9_group),
    ("med_changes", med_changes),
    ("intensity_per_day", intensity_per_day),
    ("age_mid", age_mid),
]
