"""Domain features for colleges, written from the schema and description only."""
import numpy as np
import pandas as pd


def _num(df, col):
    return pd.to_numeric(df[col], errors="coerce")


def _str(df, col):
    return df[col].astype(object).where(df[col].notna(), "").astype(str)


# Student-body affluence and early-career outcomes on log scale.
# OUTSIDE KNOWLEDGE: the AutoML-benchmark colleges target is believed to be percent of Pell-grant students, tied to family income.
def log_income_earnings(df):
    return pd.DataFrame({
        "log_faminc": np.log1p(_num(df, "faminc").clip(lower=0)),
        "log_med_earn6": np.log1p(_num(df, "median_earnings_6_years").clip(lower=0)),
        "log_med_earn10": np.log1p(_num(df, "median_earnings_10_years").clip(lower=0)),
    }, index=df.index)


# Earnings growth from year 6 to year 10 and mean-median skew (a few high earners).
def earnings_shape(df):
    m6 = _num(df, "median_earnings_6_years")
    m10 = _num(df, "median_earnings_10_years")
    a6 = _num(df, "mean_earnings_6_years")
    return pd.DataFrame({
        "earn_growth": m10 / m6.replace(0, np.nan),
        "earn6_skew": a6 / m6.replace(0, np.nan),
        "earn_to_faminc": m6 / _num(df, "faminc").replace(0, np.nan),
    }, index=df.index)


# Selectivity: combined test scores, falling back across SAT/ACT; plus admission rate.
def selectivity(df):
    sat = _num(df, "sat_total_average")
    sat2 = _num(df, "sat_verbal_midrange") + _num(df, "sat_math_midrange")
    sat = sat.fillna(sat2)
    act = _num(df, "act_combined_midrange")
    # OUTSIDE KNOWLEDGE: rough linear ACT-to-SAT (verbal+math) concordance.
    sat = sat.fillna(act * 40 + 150)
    return pd.DataFrame({
        "sat_combined": sat,
        "has_test_scores": sat.notna().astype(float),
        "admission_rate": _num(df, "admission_rate"),
    }, index=df.index)


# Cost structure: net cost (either academic or program year), out-of-state premium, and spending.
def cost_features(df):
    cost = _num(df, "average_cost_academic_year").fillna(_num(df, "average_cost_program_year"))
    tin = _num(df, "tuition_(instate)")
    tout = _num(df, "tuition_(out_of_state)")
    return pd.DataFrame({
        "cost_any": cost,
        "oos_premium": tout / tin.replace(0, np.nan),
        "spend_to_tuition": _num(df, "spend_per_student") / tin.replace(0, np.nan),
        "log_spend": np.log1p(_num(df, "spend_per_student").clip(lower=0)),
    }, index=df.index)


# Demographics: minority share and part-time / adult students (non-traditional, lower income).
def demographics(df):
    w = _num(df, "percent_white")
    return pd.DataFrame({
        "pct_black_hisp": _num(df, "percent_black") + _num(df, "percent_hispanic"),
        "pct_nonwhite": 1 - w,
        "adult_x_parttime": _num(df, "agege24") * _num(df, "percent_part_time"),
        "log_size": np.log1p(_num(df, "undergrad_size").clip(lower=0)),
    }, index=df.index)


# Institution type mapped to numbers: ownership, degree level (ordered), single-sex.
def institution_type(df):
    own = _str(df, "ownership")
    deg_map = {"Non-degree": 0, "Certificate": 1, "Associate": 2, "Bachelors": 3, "Graduate": 4}
    pd_map = {"Certificate": 1, "Associate": 2, "Bachelors": 3}
    return pd.DataFrame({
        "own_public": (own == "Public").astype(float),
        "own_forprofit": (own == "Private for-profit").astype(float),
        "highest_degree_ord": _str(df, "highest_degree").map(deg_map).astype(float),
        "predominant_degree_ord": _str(df, "predominant_degree").map(pd_map).astype(float),
        "religious": (_str(df, "religious_affiliation").replace({"None": "", "nan": ""}) != "").astype(float),
    }, index=df.index)


# Carnegie research / selectivity signals parsed from text.
def carnegie_flags(df):
    basic = _str(df, "carnegie_basic_classification").str.lower()
    ug = _str(df, "carnegie_undergraduate").str.lower()
    return pd.DataFrame({
        "research_univ": basic.str.contains("research").astype(float),
        "arts_sciences": basic.str.contains("arts & sciences").astype(float),
        "more_selective": ug.str.contains("more selective").astype(float),
        "inclusive": ug.str.contains("inclusive").astype(float),
    }, index=df.index)


# Faculty resources: salary and reliance on part-time faculty.
def faculty(df):
    return pd.DataFrame({
        "faculty_salary": _num(df, "faculty_salary"),
        "ft_faculty_share": 1 - _num(df, "percent_part_time_faculty"),
        "completion_rate": _num(df, "completion_rate"),
    }, index=df.index)


PROPOSALS = [
    ("log_income_earnings", log_income_earnings),
    ("earnings_shape", earnings_shape),
    ("selectivity", selectivity),
    ("cost_features", cost_features),
    ("demographics", demographics),
    ("institution_type", institution_type),
    ("carnegie_flags", carnegie_flags),
    ("faculty", faculty),
]
