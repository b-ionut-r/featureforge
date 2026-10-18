"""Domain features for auction, written from the schema and description only."""
import numpy as np
import pandas as pd

CAPS = ["process.b1.capacity", "process.b2.capacity", "process.b3.capacity", "process.b4.capacity"]


def _int(s):
    return pd.to_numeric(s.astype(str), errors="coerce").astype(float)


# Total remaining bidder capacity: more capacity means more possible bidding paths in the Petri net state space.
def capacity_summary(df):
    c = df[CAPS].astype(float)
    return pd.DataFrame({
        "cap_total": c.sum(axis=1),
        "n_active_bidders": (c > 0).sum(axis=1).astype(float),
        "cap_max": c.max(axis=1),
        "cap_std": c.std(axis=1),
    }, index=df.index)


# Whether a winner is already determined (0 = price not clear yet): open properties need a broader search.
def winner_known(df):
    return (_int(df["property.winner"]) > 0).astype(float)


# Price distance from the low end of the verified range (and from the reserve of 3): higher prices are deeper in the auction.
def price_depth(df):
    p = df["property.price"].astype(float)
    return pd.DataFrame({"price_above_min": p - 59, "log_price_over_reserve": np.log(p / 3.0)}, index=df.index)


# Capacity of the bidder named as winner: verifying a win by a bidder with spare capacity explores different paths.
def winner_capacity(df):
    w = _int(df["property.winner"]).fillna(0)
    c = df[CAPS].astype(float).values
    idx = w.astype(int).values
    out = np.full(len(df), np.nan)
    ok = (idx >= 1) & (idx <= 4)
    out[ok] = c[np.arange(len(df))[ok], idx[ok] - 1]
    return pd.Series(out, index=df.index)


# Product index as a number: products are verified in order, so later products may involve more explored states.
def product_index(df):
    return _int(df["property.product"])


# Interaction of price depth and capacity: a high price with lots of remaining capacity implies a long, branching auction.
def price_x_capacity(df):
    return (df["property.price"].astype(float) - 59) * df[CAPS].astype(float).sum(axis=1)


PROPOSALS = [
    ("capacity_summary", capacity_summary),
    ("winner_known", winner_known),
    ("price_depth", price_depth),
    ("winner_capacity", winner_capacity),
    ("product_index", product_index),
    ("price_x_capacity", price_x_capacity),
]
