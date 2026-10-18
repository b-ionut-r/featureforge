"""Domain features for wholesale, written from the schema and description only."""
import numpy as np
import pandas as pd

# Columns decoded from the description: V2 Fresh, V3 Milk, V4 Grocery, V5 Frozen, V6 Detergents_Paper,
# V7 Delicatessen (matches the documented first-customer spending row); V1 is a 3-level code (region-like).
CATS = ["V2", "V3", "V4", "V5", "V6", "V7"]


def total_spend(df):
    # customer size: total annual spend and its log
    t = df[CATS].sum(axis=1)
    return pd.DataFrame({"total": t, "log_total": np.log1p(t)})


def shares(df):
    # basket composition: share of each category in total spend (scale-free profile of the business)
    t = df[CATS].sum(axis=1).replace(0, np.nan)
    return pd.DataFrame({f"share_{c}": df[c] / t for c in CATS})


def retail_vs_horeca(df):
    # retailers stock packaged grocery, milk and detergents/paper; hotels/restaurants/cafes buy fresh and frozen
    # OUTSIDE KNOWLEDGE: typical purchasing patterns of retail shops vs Horeca businesses
    pack = df["V3"] + df["V4"] + df["V6"]
    fresh = df["V2"] + df["V5"]
    return pd.DataFrame({"log_pack_over_fresh": np.log1p(pack) - np.log1p(fresh),
                         "pack_share": pack / (pack + fresh).replace(0, np.nan)})


def detergent_grocery(df):
    # detergents & paper alongside grocery is the signature of a shop shelf; ratio to fresh
    return pd.DataFrame({"log_det_over_fresh": np.log1p(df["V6"]) - np.log1p(df["V2"]),
                         "log_groc_x_det": np.log1p(df["V4"]) + np.log1p(df["V6"])})


def log_spend(df):
    # spending is heavily right-skewed: log of each category
    return pd.DataFrame({f"log_{c}": np.log1p(df[c]) for c in CATS})


def diversity(df):
    # breadth of purchasing: entropy of category shares (general store vs specialised kitchen)
    p = df[CATS].div(df[CATS].sum(axis=1).replace(0, np.nan), axis=0).clip(lower=1e-9)
    return -(p * np.log(p)).sum(axis=1)


PROPOSALS = [("total_spend", total_spend), ("shares", shares), ("retail_vs_horeca", retail_vs_horeca),
             ("detergent_grocery", detergent_grocery), ("log_spend", log_spend), ("diversity", diversity)]
