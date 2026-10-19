"""Domain features for give_me_credit, written from the schema and description only."""
import numpy as np
import pandas as pd

LATE = ["NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfTimes90DaysLate"]


def _late(df):
    # 96/98 are reporting codes, not counts
    return df[LATE].astype(float).where(lambda x: x < 90)


def weighted_delinquency(df):
    # severity-weighted past-due count (90+ days counts most): the core credit-bureau risk signal
    l = _late(df)
    return l.iloc[:, 0] + 2 * l.iloc[:, 1] + 3 * l.iloc[:, 2]


def any_late_and_code(df):
    # total late events, plus a flag for the 96/98 special code (a distinct high-risk group)
    raw = df[LATE].astype(float)
    return pd.DataFrame({"late_total": _late(df).sum(axis=1, min_count=1),
                         "late_code_9x": (raw >= 90).any(axis=1).astype(float)}, index=df.index)


def monthly_debt(df):
    # DebtRatio * income = monthly debt payments; when income is missing DebtRatio already holds the amount
    inc = df["MonthlyIncome"].astype(float)
    dr = df["DebtRatio"].astype(float)
    return (dr * inc).where(inc.notna() & (inc > 0), dr)


def revolving_util_clean(df):
    # utilisation above ~1.5 is a data error, so cap it; near-1 utilisation is the danger zone
    u = df["RevolvingUtilizationOfUnsecuredLines"].astype(float)
    return u.where(u <= 1.5)


def income_per_person(df):
    # income per household member (self + dependents): spare cash to service debt
    inc = df["MonthlyIncome"].astype(float)
    dep = df["NumberOfDependents"].astype(float).fillna(0)
    return inc / (dep + 1)


def disposable_income(df):
    # income left after debt payments
    inc = df["MonthlyIncome"].astype(float)
    return inc - df["DebtRatio"].astype(float) * inc


def util_x_lines(df):
    # high utilisation spread across few open lines means little credit headroom
    u = revolving_util_clean(df)
    return u / (df["NumberOfOpenCreditLinesAndLoans"].astype(float) + 1)


def non_mortgage_lines(df):
    # open lines that are not real-estate loans (cards, personal loans)
    return (df["NumberOfOpenCreditLinesAndLoans"] - df["NumberRealEstateLoansOrLines"]).astype(float)


def late_per_age(df):
    # delinquency relative to credit history length (age as proxy)
    return weighted_delinquency(df) / (df["age"].astype(float) - 17).clip(lower=1)


PROPOSALS = [
    ("weighted_delinquency", weighted_delinquency),
    ("revolving_util_clean", revolving_util_clean),
    ("any_late_and_code", any_late_and_code),
    ("monthly_debt", monthly_debt),
    ("util_x_lines", util_x_lines),
    ("income_per_person", income_per_person),
    ("disposable_income", disposable_income),
    ("late_per_age", late_per_age),
    ("non_mortgage_lines", non_mortgage_lines),
]
