"""Domain features for coupon, written from the schema and description only."""
import numpy as np
import pandas as pd

FREQ = {"never": 0.0, "less1": 0.5, "1~3": 2.0, "4~8": 6.0, "gt8": 10.0}
COUPON_TO_HABIT = {
    "Bar": "Bar",
    "Coffee House": "CoffeeHouse",
    "Carry out & Take away": "CarryAway",
    "Restaurant(<20)": "RestaurantLessThan20",
    "Restaurant(20-50)": "Restaurant20To50",
}
HABITS = list(COUPON_TO_HABIT.values())


def _freq(s):
    return s.astype(str).str.strip().map(FREQ).astype(float)


def matching_venue_frequency(df):
    # monthly visits to the very venue type the coupon is for: the strongest acceptance signal
    out = pd.Series(np.nan, index=df.index, dtype=float)
    c = df["coupon"].astype(str)
    for cp, col in COUPON_TO_HABIT.items():
        m = c == cp
        if col in df.columns:
            out[m] = _freq(df.loc[m, col])
    return out


def distance_level(df):
    # 0 = under 15 min, 1 = 15-25 min, 2 = over 25 min detour to the venue
    a = pd.to_numeric(df["toCoupon_GEQ15min"], errors="coerce")
    b = pd.to_numeric(df["toCoupon_GEQ25min"], errors="coerce")
    return (a + b).astype(float)


def habit_x_closeness(df):
    # habitual visitors accept when it is close and on the way; penalise long detours
    f = matching_venue_frequency(df)
    d = distance_level(df)
    same = pd.to_numeric(df["direction_same"], errors="coerce").fillna(0)
    return (f + 1) / (1 + d) * (1 + same)


def total_outing_frequency(df):
    # overall eating/drinking-out propensity
    return pd.concat([_freq(df[c]) for c in HABITS if c in df.columns], axis=1).sum(axis=1, min_count=1)


def income_mid(df):
    # income band midpoint in dollars: price sensitivity for the 20-50 restaurant vs cheap venues
    s = df["income"].astype(str)
    lo = pd.to_numeric(s.str.extract(r"\$(\d+)")[0], errors="coerce")
    hi = pd.to_numeric(s.str.extract(r"-\s*\$(\d+)")[0], errors="coerce")
    mid = (lo + hi) / 2
    mid = mid.where(hi.notna(), lo)
    mid[s.str.contains("Less than", case=False)] = 6250.0
    mid[s.str.contains("or More", case=False)] = 112500.0
    return mid.astype(float)


def age_num(df):
    # numeric age (bands below21 / 50plus mapped): bars attract young, coffee houses all ages
    s = df["age"].astype(str).str.strip()
    v = pd.to_numeric(s, errors="coerce")
    v[s == "below21"] = 19.0
    v[s == "50plus"] = 55.0
    return v.astype(float)


def hours_valid(df):
    # coupon validity in hours; longer windows let people redeem later
    s = df["expiration"].astype(str).str.strip()
    return s.map({"2h": 2.0, "1d": 24.0}).astype(float)


def habit_share(df):
    # matching venue visits as share of all outings: specialised preference for this venue type
    return matching_venue_frequency(df) / (total_outing_frequency(df) + 1.0)


def driver_coupon_count(df):
    # how often the same driver profile appears: the survey shows each person many scenarios
    keys = ["gender", "age", "maritalStatus", "has_children", "education", "occupation", "income",
            "Bar", "CoffeeHouse", "CarryAway", "RestaurantLessThan20", "Restaurant20To50"]
    keys = [k for k in keys if k in df.columns]
    k = df[keys].astype(str).agg("|".join, axis=1)
    return k.map(k.value_counts()).astype(float)


PROPOSALS = [
    ("matching_venue_frequency", matching_venue_frequency),
    ("habit_x_closeness", habit_x_closeness),
    ("habit_share", habit_share),
    ("distance_level", distance_level),
    ("total_outing_frequency", total_outing_frequency),
    ("income_mid", income_mid),
    ("age_num", age_num),
    ("hours_valid", hours_valid),
    ("driver_coupon_count", driver_coupon_count),
]
