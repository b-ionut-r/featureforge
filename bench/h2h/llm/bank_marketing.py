"""Domain features for bank_marketing, written from the schema and description only.
Columns are the UCI bank-marketing fields in order: V1 age, V2 job, V3 marital, V4 education, V5 default,
V6 balance, V7 housing, V8 loan, V9 contact, V10 day, V11 month, V12 duration, V13 campaign, V14 pdays,
V15 previous, V16 poutcome."""
import numpy as np
import pandas as pd

MONTHS = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct",
                                          "nov", "dec"])}


def _n(df, c):
    return pd.to_numeric(df[c], errors="coerce").astype(float)


def _mon(df):
    return df["V11"].astype("string").str.strip().str.lower().map(MONTHS).astype(float)


def calls_same_day(df):
    # call volume on that calendar day: mass-campaign days (huge volume, May) convert poorly, quiet days convert well
    key = df["V11"].astype(str) + "_" + df["V10"].astype(str)
    return key.groupby(key).transform("size").astype(float)


def day_of_year(df):
    # calendar position of the contact; campaign phases and seasonality drive success rates
    return (_mon(df) - 1) * 30.5 + _n(df, "V10")


def log_duration(df):
    # last call duration is the strongest indicator of interest; log tames its skew
    return np.log1p(_n(df, "V12").clip(lower=0))


def duration_per_campaign(df):
    # long talk with few attempts = genuine interest, many short attempts = reluctance
    return _n(df, "V12") / _n(df, "V13").replace(0, np.nan)


def previously_contacted(df):
    # pdays=-1 means never contacted in a previous campaign; recency of earlier contact matters
    p = _n(df, "V14")
    return pd.DataFrame({"was_contacted": (p >= 0).astype(float), "pdays_clean": p.where(p >= 0)})


def duration_vs_month(df):
    # duration relative to that month's typical call length (call scripts / campaign phase differ)
    d = _n(df, "V12")
    return d / d.groupby(df["V11"].astype(str)).transform("median").replace(0, np.nan)


def balance_vs_job(df):
    # wealth relative to occupation peers: spare cash for a term deposit
    b = _n(df, "V6")
    return b - b.groupby(df["V2"].astype(str)).transform("median")


def debt_count(df):
    # number of credit obligations (default, housing loan, personal loan) reduces ability to save
    return sum((df[c].astype("string").str.strip().str.lower() == "yes").astype(float) for c in ["V5", "V7", "V8"])


def prev_success_rate(df):
    # contacts per previous campaign attempt; prior successes are very predictive
    pout = (df["V16"].astype("string").str.strip() == "success").astype(float)
    return pout * 10 + _n(df, "V15") / (_n(df, "V13") + _n(df, "V15")).replace(0, np.nan)


PROPOSALS = [("calls_same_day", calls_same_day), ("log_duration", log_duration),
             ("duration_per_campaign", duration_per_campaign), ("day_of_year", day_of_year),
             ("previously_contacted", previously_contacted), ("duration_vs_month", duration_vs_month),
             ("prev_success_rate", prev_success_rate), ("balance_vs_job", balance_vs_job), ("debt_count", debt_count)]
