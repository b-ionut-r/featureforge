"""Domain features for t10, written from the schema and description only."""
import numpy as np
import pandas as pd

# OUTSIDE KNOWLEDGE: the columns match the UCI student-performance (Portuguese course, 649 students) table:
# f_1 school, f_2 sex, f_3 age, f_4 address, f_5 famsize, f_6 Pstatus, f_7 Medu, f_8 Fedu, f_9 Mjob, f_10 Fjob,
# f_11 reason, f_12 guardian, f_13 traveltime, f_14 studytime, f_15 failures, f_16 schoolsup, f_17 famsup,
# f_18 paid, f_19 activities, f_20 nursery, f_21 higher, f_22 internet, f_23 romantic, f_24 famrel,
# f_25 freetime, f_26 goout, f_27 Dalc, f_28 Walc, f_29 health, f_30 absences. Target likely the final grade.


def _n(df, c):
    return pd.to_numeric(df[c], errors="coerce")


def _yes(df, c):
    return (df[c].astype(object).astype(str).str.strip().str.lower() == "yes").astype(float)


# Parental education: combined/maximum level, a strong socioeconomic predictor of achievement.
def parent_education(df):
    m = _n(df, "f_7")
    f = _n(df, "f_8")
    return pd.DataFrame({
        "pedu_sum": m + f,
        "pedu_max": np.maximum(m, f),
        "pedu_diff": m - f,
        "both_higher_ed": ((m == 4) & (f == 4)).astype(float),
    })


# Alcohol use: weighted weekly consumption (workday 5 days + weekend 2 days) and heavy-use flag.
def alcohol(df):
    d = _n(df, "f_27")
    w = _n(df, "f_28")
    return pd.DataFrame({
        "alc_weekly": (5 * d + 2 * w) / 7.0,
        "alc_heavy": ((d >= 3) | (w >= 4)).astype(float),
        "alc_x_goout": (d + w) * _n(df, "f_26"),
    })


# Academic engagement: study time vs leisure, past failures, ambition for higher education, absences.
def engagement(df):
    study = _n(df, "f_14")
    free = _n(df, "f_25")
    goout = _n(df, "f_26")
    fail = _n(df, "f_15")
    absn = _n(df, "f_30")
    higher = _yes(df, "f_21")
    return pd.DataFrame({
        "study_minus_leisure": study - (free + goout) / 2.0,
        "any_failure": (fail > 0).astype(float),
        "log_absences": np.log1p(absn.clip(lower=0)),
        "no_higher_ambition": 1 - higher,
        "engagement_score": study - fail + higher - np.log1p(absn.clip(lower=0)) / 2,
    })


# Support resources: count of extra supports and home resources (internet, nursery, family support, paid classes).
def support(df):
    cols = ["f_16", "f_17", "f_18", "f_20", "f_22"]
    y = pd.DataFrame({c: _yes(df, c) for c in cols})
    return pd.DataFrame({
        "n_supports": y.sum(axis=1),
        "school_support_flag": y["f_16"],
        "home_resources": y[["f_20", "f_22"]].sum(axis=1) + _yes(df, "f_17"),
    })


# Age relative to grade: students older than typical for secondary school have usually repeated years.
def age_lag(df):
    age = _n(df, "f_3")
    fail = _n(df, "f_15")
    return pd.DataFrame({
        "over_age": (age - 17).clip(lower=0),
        "age_x_fail": age * fail,
        "age_ge_19": (age >= 19).astype(float),
    })


# Family context: parents' occupations as skill-level ordinals, family relations, cohabitation, guardian.
def family_context(df):
    job = {"at_home": 0, "other": 1, "services": 2, "health": 3, "teacher": 3}
    mj = df["f_9"].astype(object).map(job).astype(float)
    fj = df["f_10"].astype(object).map(job).astype(float)
    return pd.DataFrame({
        "parent_job_sum": mj + fj,
        "mother_teacher_or_health": df["f_9"].astype(object).isin(["teacher", "health"]).astype(float),
        "parents_apart": (df["f_6"].astype(object) == "A").astype(float),
        "guardian_other": (df["f_12"].astype(object) == "other").astype(float),
        "famrel_low": (_n(df, "f_24") <= 2).astype(float),
    })


# School and location: rural, long commute, school identity, and reason-for-school choice (reputation).
def school_location(df):
    rural = (df["f_4"].astype(object) == "R").astype(float)
    travel = _n(df, "f_13")
    return pd.DataFrame({
        "school_ms": (df["f_1"].astype(object) == "MS").astype(float),
        "rural_long_commute": rural * (travel >= 2).astype(float),
        "reason_reputation": (df["f_11"].astype(object) == "reputation").astype(float),
        "commute_minus_study": travel - _n(df, "f_14"),
    })


PROPOSALS = [
    ("parent_education", parent_education),
    ("alcohol", alcohol),
    ("engagement", engagement),
    ("support", support),
    ("age_lag", age_lag),
    ("family_context", family_context),
    ("school_location", school_location),
]
