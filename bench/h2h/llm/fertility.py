"""Domain features for fertility, written from the schema and description only."""
import numpy as np
import pandas as pd

# Columns decoded from the description's attribute list (in order): V1 season (-1 winter .. 1 fall), V2 age (scaled 18-36),
# V3 childish diseases (0 yes, 1 no), V4 accident/trauma (0 yes, 1 no), V5 surgery (0 yes, 1 no),
# V6 high fever (-1 <3 months ago, 0 >3 months ago, 1 none), V7 alcohol frequency (higher = less often),
# V8 smoking (-1 never, 0 occasional, 1 daily), V9 hours sitting per day (scaled).


def lifestyle_risk(df):
    # lifestyle insults to spermatogenesis: frequent alcohol, daily smoking, long sitting (scrotal heat)
    alcohol = 1.0 - df["V7"]
    smoke = (df["V8"] + 1.0) / 2.0
    return pd.DataFrame({"lifestyle_sum": alcohol + smoke + df["V9"], "alcohol_freq": alcohol, "smoke_level": smoke})


def medical_history(df):
    # count of past health insults: childhood disease (e.g. mumps orchitis), trauma, surgery, recent fever
    s = (1 - df["V3"]) + (1 - df["V4"]) + (1 - df["V5"]) + (df["V6"] < 1).astype(float)
    return s.astype(float)


def recent_fever(df):
    # high fever within the last 3 months disrupts the ~74-day sperm production cycle
    # OUTSIDE KNOWLEDGE: spermatogenesis takes about 2.5 months, so recent fevers affect the current sample
    return pd.DataFrame({"fever_recent": (df["V6"] == -1).astype(float), "fever_any": (df["V6"] < 1).astype(float)})


def season_cyclic(df):
    # season of analysis: summer heat lowers sperm quality; encode cyclically
    idx = ((df["V1"] + 1.0) / 0.67).round()  # 0 winter, 1 spring, 2 summer, 3 fall
    ang = 2 * np.pi * idx / 4.0
    return pd.DataFrame({"season_sin": np.sin(ang), "season_cos": np.cos(ang), "summer": (idx == 2).astype(float)})


def age_x_sitting(df):
    # older age combined with sedentary habits
    return pd.DataFrame({"age_x_sitting": df["V2"] * df["V9"], "age_x_smoke": df["V2"] * (df["V8"] + 1.0)})


def total_risk(df):
    # combined lifestyle + medical risk index
    return (medical_history(df) + lifestyle_risk(df)["lifestyle_sum"]).astype(float)


PROPOSALS = [("lifestyle_risk", lifestyle_risk), ("medical_history", medical_history), ("recent_fever", recent_fever),
             ("season_cyclic", season_cyclic), ("age_x_sitting", age_x_sitting), ("total_risk", total_risk)]
