"""Domain features for t07, written from the schema and description only."""
import numpy as np
import pandas as pd

# OUTSIDE KNOWLEDGE: the columns match the IBM Telco customer-churn table: f_1 gender, f_2 senior citizen,
# f_3 partner, f_4 dependents, f_5 tenure (months), f_6 phone service, f_7 multiple lines, f_8 internet service,
# f_9..f_14 online security/backup/device protection/tech support/streaming TV/streaming movies, f_15 contract,
# f_16 paperless billing, f_17 payment method, f_18 monthly charges, f_19 total charges (string, blanks for new).

_ADDONS = ["f_9", "f_10", "f_11", "f_12", "f_13", "f_14"]


def _s(df, c):
    return df[c].astype(object).astype(str).str.strip()


def _total(df):
    return pd.to_numeric(_s(df, "f_19").replace("", np.nan), errors="coerce")


# Billing consistency: total vs tenure*monthly reveals price changes; average historical charge per month.
def billing(df):
    tenure = pd.to_numeric(df["f_5"], errors="coerce")
    monthly = pd.to_numeric(df["f_18"], errors="coerce")
    total = _total(df).fillna(0)
    avg = total / tenure.replace(0, np.nan)
    return pd.DataFrame({
        "total_charges_num": total,
        "avg_monthly_hist": avg,
        "monthly_minus_hist": monthly - avg,
        "total_over_expected": total / (tenure * monthly).replace(0, np.nan),
        "log_total": np.log1p(total.clip(lower=0)),
    })


# Service bundle: number of add-ons and protective (security/backup/protection/support) vs streaming services.
def service_counts(df):
    yes = pd.DataFrame({c: (_s(df, c) == "Yes").astype(float) for c in _ADDONS})
    phone = (_s(df, "f_6") == "Yes").astype(float)
    multi = (_s(df, "f_7") == "Yes").astype(float)
    internet = (_s(df, "f_8") != "No").astype(float)
    return pd.DataFrame({
        "n_addons": yes.sum(axis=1),
        "n_protective": yes[["f_9", "f_10", "f_11", "f_12"]].sum(axis=1),
        "n_streaming": yes[["f_13", "f_14"]].sum(axis=1),
        "n_services_total": yes.sum(axis=1) + phone + multi + internet,
        "fiber_no_protection": ((_s(df, "f_8") == "Fiber optic") &
                                (yes[["f_9", "f_10", "f_11", "f_12"]].sum(axis=1) == 0)).astype(float),
    })


# Commitment: ordered contract length, automatic payment, electronic check (known high-churn payment).
def commitment(df):
    contract = _s(df, "f_15").map({"Month-to-month": 0, "One year": 1, "Two year": 2}).astype(float)
    pay = _s(df, "f_17")
    return pd.DataFrame({
        "contract_ord": contract,
        "auto_pay": pay.str.contains("automatic", case=False).astype(float),
        "electronic_check": (pay == "Electronic check").astype(float),
        "mtm_paperless_echeck": ((contract == 0) & (_s(df, "f_16") == "Yes") &
                                 (pay == "Electronic check")).astype(float),
    })


# Tenure stage: new customers churn most; tenure relative to contract length and to remaining commitment.
def tenure_stage(df):
    tenure = pd.to_numeric(df["f_5"], errors="coerce")
    contract_m = _s(df, "f_15").map({"Month-to-month": 1, "One year": 12, "Two year": 24}).astype(float)
    return pd.DataFrame({
        "log_tenure": np.log1p(tenure.clip(lower=0)),
        "tenure_le_6": (tenure <= 6).astype(float),
        "tenure_years": np.floor(tenure / 12),
        "months_into_contract": np.mod(tenure, contract_m),
    })


# Price per service: monthly charge divided by the number of services held (perceived value for money).
def price_per_service(df):
    monthly = pd.to_numeric(df["f_18"], errors="coerce")
    yes = sum((_s(df, c) == "Yes").astype(float) for c in _ADDONS)
    n = yes + (_s(df, "f_6") == "Yes").astype(float) + (_s(df, "f_7") == "Yes").astype(float) + \
        (_s(df, "f_8") != "No").astype(float)
    internet = _s(df, "f_8").map({"No": 0, "DSL": 1, "Fiber optic": 2}).astype(float)
    return pd.DataFrame({
        "charge_per_service": monthly / n.replace(0, np.nan),
        "internet_ord": internet,
        "monthly_x_mtm": monthly * (_s(df, "f_15") == "Month-to-month").astype(float),
    })


# Household: partner and dependents signal stability; senior citizens living alone churn differently.
def household(df):
    partner = (_s(df, "f_3") == "Yes").astype(float)
    dep = (_s(df, "f_4") == "Yes").astype(float)
    senior = pd.to_numeric(df["f_2"], errors="coerce")
    return pd.DataFrame({
        "family_size_proxy": partner + dep,
        "senior_alone": ((senior == 1) & (partner == 0) & (dep == 0)).astype(float),
    })


PROPOSALS = [
    ("billing", billing),
    ("service_counts", service_counts),
    ("commitment", commitment),
    ("tenure_stage", tenure_stage),
    ("price_per_service", price_per_service),
    ("household", household),
]
