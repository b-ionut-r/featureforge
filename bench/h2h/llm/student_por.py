"""Domain features for student_por, written from the schema and description only."""
import numpy as np
import pandas as pd


def _yes(s):
    return (s.astype(str).str.lower() == "yes").astype(float)


# Parental education: combined family cultural capital.
def parent_education(df):
    m = df["Medu"].astype(float)
    f = df["Fedu"].astype(float)
    return pd.DataFrame({"pedu_mean": (m + f) / 2, "pedu_max": np.maximum(m, f)}, index=df.index)


# Weekly alcohol use: 5 weekdays + 2 weekend days weighting.
def weekly_alcohol(df):
    return (5 * df["Dalc"].astype(float) + 2 * df["Walc"].astype(float)) / 7


# Academic risk: past failures dominate, plus a binary any-failure flag.
def failure_flags(df):
    f = df["failures"].astype(float)
    return pd.DataFrame({"any_failure": (f > 0).astype(float), "failures_sq": f ** 2}, index=df.index)


# Study effort versus social distraction.
def study_vs_social(df):
    return df["studytime"].astype(float) - (df["goout"].astype(float) + df["freetime"].astype(float)) / 2


# Count of educational supports and resources at hand.
def support_count(df):
    return _yes(df["schoolsup"]) + _yes(df["famsup"]) + _yes(df["paid"]) + _yes(df["internet"]) + _yes(df["nursery"])


# Motivation: wants higher education, combined with study time.
def motivation(df):
    return _yes(df["higher"]) * df["studytime"].astype(float)


# Absences are long-tailed; log scale.
def log_absences(df):
    return np.log1p(df["absences"].astype(float))


# Parent with an education-related job (teacher/health) signals academic environment.
def parent_professional_job(df):
    prof = {"teacher", "health"}
    m = df["Mjob"].astype(str).isin(prof).astype(float)
    f = df["Fjob"].astype(str).isin(prof).astype(float)
    return m + f


# School and demographic binaries mapped to numbers.
def binaries(df):
    return pd.DataFrame({
        "school_MS": (df["school"].astype(str) == "MS").astype(float),
        "male": (df["sex"].astype(str) == "M").astype(float),
        "urban": (df["address"].astype(str) == "U").astype(float),
        "reason_reputation": (df["reason"].astype(str) == "reputation").astype(float),
        "romantic": _yes(df["romantic"]),
    }, index=df.index)


# Older than typical for the grade suggests repeated years.
def over_age(df):
    return (df["age"].astype(float) - 17).clip(lower=0)


PROPOSALS = [
    ("parent_education", parent_education),
    ("weekly_alcohol", weekly_alcohol),
    ("failure_flags", failure_flags),
    ("study_vs_social", study_vs_social),
    ("support_count", support_count),
    ("motivation", motivation),
    ("log_absences", log_absences),
    ("parent_professional_job", parent_professional_job),
    ("binaries", binaries),
    ("over_age", over_age),
]
