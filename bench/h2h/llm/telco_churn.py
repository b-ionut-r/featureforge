"""Domain features for telco_churn, written from the schema and description only."""
import numpy as np
import pandas as pd

ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]


def _yes(s):
    return (s.astype(str).str.strip() == "Yes").astype(float)


def _total(df):
    return pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")


# Contract length as ordered months: longer commitments lock customers in.
def contract_months(df):
    m = {"Month-to-month": 1.0, "One year": 12.0, "Two year": 24.0}
    return df["Contract"].astype(str).map(m).astype(float)


# Number of add-on and protective services: more services means more switching cost.
def service_counts(df):
    sec = df[["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]].apply(_yes).sum(axis=1)
    stream = df[["StreamingTV", "StreamingMovies"]].apply(_yes).sum(axis=1)
    total = df[ADDONS].apply(_yes).sum(axis=1) + _yes(df["PhoneService"]) + _yes(df["MultipleLines"])
    return pd.DataFrame({"n_protect": sec, "n_stream": stream, "n_services": total}, index=df.index)


# Average historical monthly bill vs current bill: a recent price increase can trigger churn.
def charge_trend(df):
    tot = _total(df)
    ten = df["tenure"].astype(float)
    avg = tot / ten.replace(0, np.nan)
    return pd.DataFrame({
        "avg_monthly_hist": avg,
        "monthly_minus_hist": df["MonthlyCharges"] - avg,
        "total_minus_expected": tot - df["MonthlyCharges"] * ten,
    }, index=df.index)


# Price paid per service: high price for few services feels like poor value.
def price_per_service(df):
    n = df[ADDONS].apply(_yes).sum(axis=1) + _yes(df["PhoneService"]) + _yes(df["MultipleLines"]) \
        + (df["InternetService"].astype(str) != "No").astype(float)
    return df["MonthlyCharges"] / n.replace(0, np.nan)


# Friction-free payment and billing: automatic payment reduces lapse, electronic check is a common churn signal.
def payment_flags(df):
    pm = df["PaymentMethod"].astype(str)
    return pd.DataFrame({
        "auto_pay": pm.str.contains("automatic").astype(float),
        "electronic_check": (pm == "Electronic check").astype(float),
        "paperless": _yes(df["PaperlessBilling"]),
    }, index=df.index)


# Fiber customers without support/security: expensive, unsupported service is a classic dissatisfaction combo.
def fiber_unprotected(df):
    fiber = (df["InternetService"].astype(str) == "Fiber optic").astype(float)
    nosup = 1.0 - _yes(df["TechSupport"])
    nosec = 1.0 - _yes(df["OnlineSecurity"])
    return pd.DataFrame({"fiber": fiber, "fiber_no_support": fiber * nosup * nosec}, index=df.index)


# Household ties: partner/dependents indicate stability; senior alone is a distinct segment.
def household(df):
    p = _yes(df["Partner"])
    d = _yes(df["Dependents"])
    s = df["SeniorCitizen"].astype(float)
    return pd.DataFrame({"family_size": p + d, "senior_alone": s * (1 - p) * (1 - d)}, index=df.index)


# Early-tenure month-to-month customers are the riskiest; tenure relative to contract length.
def tenure_contract(df):
    m = df["Contract"].astype(str).map({"Month-to-month": 1.0, "One year": 12.0, "Two year": 24.0}).astype(float)
    ten = df["tenure"].astype(float)
    return pd.DataFrame({
        "log_tenure": np.log1p(ten),
        "new_mtm": ((ten <= 6) & (m == 1)).astype(float),
        "tenure_over_contract": ten / m.replace(0, np.nan),
    }, index=df.index)


PROPOSALS = [
    ("contract_months", contract_months),
    ("service_counts", service_counts),
    ("charge_trend", charge_trend),
    ("price_per_service", price_per_service),
    ("payment_flags", payment_flags),
    ("fiber_unprotected", fiber_unprotected),
    ("household", household),
    ("tenure_contract", tenure_contract),
]
