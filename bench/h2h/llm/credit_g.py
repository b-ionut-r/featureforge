"""Domain features for credit_g, written from the schema and description only."""
import numpy as np
import pandas as pd

CHECK = {"<0": 0.0, "0<=X<200": 1.0, ">=200": 2.0, "no checking": 3.0}
SAV = {"<100": 0.0, "100<=X<500": 1.0, "500<=X<1000": 2.0, ">=1000": 3.0, "no known savings": np.nan}
EMP = {"unemployed": 0.0, "<1": 0.5, "1<=X<4": 2.5, "4<=X<7": 5.5, ">=7": 9.0}


def _s(df, c):
    return df[c].astype(str).str.strip()


def monthly_payment(df):
    # credit amount per month of duration: the actual repayment burden
    return df["credit_amount"].astype(float) / df["duration"].astype(float).replace(0, np.nan)


def implied_disposable_income(df):
    # installment rate is % of disposable income, so payment / rate scales with income
    return monthly_payment(df) / df["installment_commitment"].astype(float).replace(0, np.nan)


def liquidity_score(df):
    # checking + savings level (ordinal); no savings with negative checking is the classic bad profile
    c = _s(df, "checking_status").map(CHECK).astype(float)
    s = _s(df, "savings_status").map(SAV).astype(float)
    return c.fillna(1) + s.fillna(0.5)


def amount_vs_purpose(df):
    # loan size relative to typical loans for the same purpose
    a = df["credit_amount"].astype(float)
    return a / a.groupby(_s(df, "purpose")).transform("median")


def amount_per_age(df):
    # large loans to young applicants are riskier
    return df["credit_amount"].astype(float) / df["age"].astype(float).replace(0, np.nan)


def employment_years_ratio(df):
    # job tenure relative to adult life: stability signal
    e = _s(df, "employment").map(EMP).astype(float)
    return e / (df["age"].astype(float) - 18).clip(lower=1)


def duration_x_rate(df):
    # long loans that also take a big share of income compound repayment risk
    return df["duration"].astype(float) * df["installment_commitment"].astype(float)


def is_female(df):
    # sex parsed from the combined personal_status string
    return _s(df, "personal_status").str.startswith("female").astype(float)


PROPOSALS = [
    ("monthly_payment", monthly_payment),
    ("liquidity_score", liquidity_score),
    ("amount_vs_purpose", amount_vs_purpose),
    ("implied_disposable_income", implied_disposable_income),
    ("duration_x_rate", duration_x_rate),
    ("amount_per_age", amount_per_age),
    ("employment_years_ratio", employment_years_ratio),
    ("is_female", is_female),
]
