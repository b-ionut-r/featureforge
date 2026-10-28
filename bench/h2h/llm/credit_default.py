"""Domain features for credit_default, written from the schema and description only."""
import numpy as np
import pandas as pd

PAY = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL = [f"BILL_AMT{i}" for i in range(1, 7)]
PAMT = [f"PAY_AMT{i}" for i in range(1, 7)]


def _lim(df):
    return df["LIMIT_BAL"].astype(float).replace(0, np.nan)


def utilization_latest(df):
    # latest statement balance over credit limit: maxed-out cards default far more often
    return df["BILL_AMT1"].astype(float) / _lim(df)


def repayment_ratio_latest(df):
    # share of the previous statement actually repaid this month (PAY_AMT1 settles BILL_AMT2)
    b = df["BILL_AMT2"].astype(float)
    return (df["PAY_AMT1"].astype(float) / b.where(b > 0)).clip(upper=5)


def delay_summary(df):
    # recent and chronic delinquency: worst delay, months delinquent, and trend in status
    p = df[PAY].astype(float)
    return pd.DataFrame({
        "pay_max_delay": p.max(axis=1),
        "pay_n_delayed": (p > 0).sum(axis=1).astype(float),
        "pay_sum_delay": p.clip(lower=0).sum(axis=1),
        "pay_trend": df["PAY_0"].astype(float) - df["PAY_6"].astype(float),
    }, index=df.index)


def total_repaid_over_billed(df):
    # six-month repayment coverage: paid amount over billed amount (payments m settle bill m+1)
    paid = df[PAMT[:5]].astype(float).sum(axis=1)
    billed = df[BILL[1:]].astype(float).clip(lower=0).sum(axis=1)
    return (paid / billed.replace(0, np.nan)).clip(upper=5)


def mean_utilization(df):
    # average balance over limit across six months, plus peak utilisation
    b = df[BILL].astype(float)
    lim = _lim(df)
    return pd.DataFrame({"util_mean": b.mean(axis=1) / lim, "util_max": b.max(axis=1) / lim}, index=df.index)


def bill_growth(df):
    # balance growth from six months ago to now, relative to limit: debt is spiralling
    return (df["BILL_AMT1"].astype(float) - df["BILL_AMT6"].astype(float)) / _lim(df)


def headroom(df):
    # remaining available credit in dollars
    return df["LIMIT_BAL"].astype(float) - df["BILL_AMT1"].astype(float)


def n_zero_payments(df):
    # months with no payment at all while a balance was owed
    paid = df[PAMT[:5]].astype(float).to_numpy()
    owed = df[BILL[1:]].astype(float).to_numpy()
    return pd.Series(((paid == 0) & (owed > 0)).sum(axis=1).astype(float), index=df.index)


def limit_vs_age_group(df):
    # credit limit relative to peers of the same age: banks grant less to riskier profiles
    age_band = (df["AGE"].astype(float) // 5)
    lim = df["LIMIT_BAL"].astype(float)
    return lim / lim.groupby(age_band).transform("median").replace(0, np.nan)


PROPOSALS = [
    ("delay_summary", delay_summary),
    ("repayment_ratio_latest", repayment_ratio_latest),
    ("utilization_latest", utilization_latest),
    ("total_repaid_over_billed", total_repaid_over_billed),
    ("mean_utilization", mean_utilization),
    ("n_zero_payments", n_zero_payments),
    ("bill_growth", bill_growth),
    ("headroom", headroom),
    ("limit_vs_age_group", limit_vs_age_group),
]
