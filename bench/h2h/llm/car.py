"""Domain features for car, written from the schema and description only."""
import numpy as np
import pandas as pd

_PRICE = {"low": 0, "med": 1, "high": 2, "vhigh": 3}
_DOORS = {"2": 2, "3": 3, "4": 4, "5more": 5}
_PERS = {"2": 2, "4": 4, "more": 6}
_LUG = {"small": 0, "med": 1, "big": 2}
_SAFE = {"low": 0, "med": 1, "high": 2}


def _o(df):
    return pd.DataFrame({"buying": df["buying"].astype(str).map(_PRICE).astype(float),
                         "maint": df["maint"].astype(str).map(_PRICE).astype(float),
                         "doors": df["doors"].astype(str).map(_DOORS).astype(float),
                         "persons": df["persons"].astype(str).map(_PERS).astype(float),
                         "lug_boot": df["lug_boot"].astype(str).map(_LUG).astype(float),
                         "safety": df["safety"].astype(str).map(_SAFE).astype(float)}, index=df.index)


def ordinals(df):
    # the six attributes are ordered levels; numeric codes let a model use the order
    return _o(df).add_suffix("_ord")


def price_concept(df):
    # PRICE intermediate concept = buying + maintenance cost (lower is better)
    o = _o(df)
    return pd.DataFrame({"price_sum": o["buying"] + o["maint"], "price_max": o[["buying", "maint"]].max(axis=1)})


def comfort_concept(df):
    # COMFORT intermediate concept combines doors, persons and luggage boot
    o = _o(df)
    return pd.DataFrame({"comfort_sum": (o["doors"] - 2) / 3 + (o["persons"] - 2) / 4 + o["lug_boot"] / 2,
                         "comfort_min": pd.concat([(o["persons"] - 2) / 4, o["lug_boot"] / 2], axis=1).min(axis=1)})


def tech_concept(df):
    # TECH = comfort and safety; weakest-link aggregation mirrors how DEX rules usually behave
    o = _o(df)
    comfort = ((o["persons"] - 2) / 4 + o["lug_boot"] / 2 + (o["doors"] - 2) / 3) / 3
    return pd.DataFrame({"tech_min": np.minimum(comfort * 2, o["safety"]), "tech_sum": comfort * 2 + o["safety"]})


# OUTSIDE KNOWLEDGE: in the published DEX car model, persons=2 or safety=low makes a car unacceptable regardless of other attributes.
def knockout(df):
    # hard veto attributes: only two seats or low safety
    o = _o(df)
    return ((o["persons"] == 2) | (o["safety"] == 0)).astype(float)


def overall_score(df):
    # crude utility: good tech minus high price
    o = _o(df)
    return o["safety"] + o["lug_boot"] / 2 + (o["persons"] - 2) / 4 - (o["buying"] + o["maint"]) / 3


def big_cheap_safe(df):
    # interactions: high safety but expensive, many persons but small boot (crowded)
    o = _o(df)
    return pd.DataFrame({"safety_x_cheap": o["safety"] * (3 - o["buying"]),
                         "persons_vs_doors": o["persons"] - o["doors"],
                         "persons_x_lug": o["persons"] * (o["lug_boot"] + 1)})


PROPOSALS = [("ordinals", ordinals), ("price_concept", price_concept), ("comfort_concept", comfort_concept),
             ("tech_concept", tech_concept), ("knockout", knockout), ("overall_score", overall_score),
             ("big_cheap_safe", big_cheap_safe)]
