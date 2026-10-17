"""Domain features for bank_churn, written from the schema and description only."""
import numpy as np
import pandas as pd


def _b(df, c):
    return pd.to_numeric(df[c], errors="coerce").astype(float)


def balance_to_salary(df):
    # customers parking a large share of their income here are more engaged (or more at risk of moving it)
    return _b(df, "balance") / _b(df, "estimated_salary").replace(0, np.nan)


def products_x_active(df):
    # engagement profile: 1 product inactive and 3-4 products are the classic churn segments
    return _b(df, "products_number") * 2 + _b(df, "active_member")


def age_x_inactive(df):
    # older inactive customers churn the most (retirement / consolidating banking)
    return _b(df, "age") * (1 - _b(df, "active_member"))


def balance_per_product(df):
    # money held per product: high balance spread on one product indicates a savings-only relationship
    return _b(df, "balance") / _b(df, "products_number").replace(0, np.nan)


def tenure_over_age(df):
    # share of adult life spent with the bank: loyalty relative to age
    return _b(df, "tenure") / (_b(df, "age") - 17).clip(lower=1)


def balance_vs_country(df):
    # balance relative to country peers (Germany has very different balance distribution)
    b = _b(df, "balance")
    return b - b.groupby(df["country"].astype(str)).transform("mean")


def zero_balance(df):
    # zero balance = transactional or dormant account, different churn behaviour
    return (_b(df, "balance") == 0).astype(float)


def credit_score_vs_age(df):
    # credit score relative to customers of the same age band
    cs = _b(df, "credit_score")
    ab = (_b(df, "age") // 5)
    return cs - cs.groupby(ab).transform("mean")


PROPOSALS = [("products_x_active", products_x_active), ("age_x_inactive", age_x_inactive),
             ("balance_to_salary", balance_to_salary), ("balance_per_product", balance_per_product),
             ("tenure_over_age", tenure_over_age), ("balance_vs_country", balance_vs_country),
             ("zero_balance", zero_balance), ("credit_score_vs_age", credit_score_vs_age)]
