"""Domain features for delta_elevators, written from the schema and description only."""
import numpy as np
import pandas as pd


def climb_next(df):
    # extrapolated climb rate using its first and second differences
    return df["climbRate"] + df["diffClb"] + 0.5 * df["diffDiffClb"]


def climb_vs_altitude(df):
    # altitude error and climb rate combine like a PD controller on altitude
    return pd.DataFrame({"alt_plus_climb": df["Altitude"] + 5 * df["climbRate"], "alt_x_climb": df["Altitude"] * df["climbRate"]})


def roll_terms(df):
    # banked flight loses lift; roll and roll rate interact with elevator demand
    return pd.DataFrame({"roll_x_rate": df["curRoll"] * df["RollRate"], "abs_roll": df["curRoll"].abs(),
                         "roll_pred": df["curRoll"] + 10 * df["RollRate"]})


def climb_accel_ratio(df):
    # relative change of climb rate
    return df["diffClb"] / df["climbRate"].abs().replace(0, np.nan).clip(lower=0.1)


def bank_climb(df):
    # turn-induced sink: climb rate interacting with bank magnitude
    return df["climbRate"] * df["curRoll"].abs()


PROPOSALS = [("climb_next", climb_next), ("climb_vs_altitude", climb_vs_altitude), ("roll_terms", roll_terms),
             ("climb_accel_ratio", climb_accel_ratio), ("bank_climb", bank_climb)]
