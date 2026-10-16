"""Domain features for churn, written from the schema and description only."""
import numpy as np
import pandas as pd


def _n(df, c):
    return pd.to_numeric(df[c], errors="coerce").astype(float)


def total_charge(df):
    # the monthly bill is what drives telecom churn: heavy users paying a lot leave for cheaper plans
    return sum(_n(df, f"total {p} charge") for p in ["day", "eve", "night", "intl"])


def total_minutes(df):
    # total usage across all periods
    return sum(_n(df, f"total {p} minutes") for p in ["day", "eve", "night", "intl"])


def service_calls_high(df):
    # 4+ customer service calls is the classic unresolved-complaint churn trigger
    return (_n(df, "number customer service calls") >= 4).astype(float)


def intl_plan_x_minutes(df):
    # international plan holders with low intl usage pay for nothing; without the plan, heavy intl users overpay
    return _n(df, "international plan") * 100 - _n(df, "total intl minutes") * (1 - 2 * _n(df, "international plan"))


def intl_calls_short(df):
    # few intl calls (<3) with the plan: plan isn't worth it
    return _n(df, "total intl calls") * (1 + _n(df, "international plan") * 10)


def charge_x_service_calls(df):
    # high bill combined with complaints
    return total_charge(df) * _n(df, "number customer service calls")


def total_calls(df):
    # total number of calls, engagement
    return sum(_n(df, f"total {p} calls") for p in ["day", "eve", "night", "intl"])


def day_charge_no_vmail(df):
    # high daytime spend without voicemail plan (no bundle benefits)
    return _n(df, "total day charge") * (1 - _n(df, "voice mail plan"))


PROPOSALS = [("total_charge", total_charge), ("service_calls_high", service_calls_high),
             ("charge_x_service_calls", charge_x_service_calls), ("intl_plan_x_minutes", intl_plan_x_minutes),
             ("intl_calls_short", intl_calls_short), ("total_minutes", total_minutes),
             ("day_charge_no_vmail", day_charge_no_vmail), ("total_calls", total_calls)]
