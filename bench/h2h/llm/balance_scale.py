"""Domain features for balance_scale, written from the schema and description only."""
import numpy as np
import pandas as pd


def torque_diff(df):
    # the scale tips toward the larger torque (weight x distance); the sign of the difference is the class
    return df["left-weight"] * df["left-distance"] - df["right-weight"] * df["right-distance"]


def torque_sign(df):
    # -1/0/+1 directly encodes right / balanced / left
    return np.sign(torque_diff(df)).astype(float)


def log_torque_ratio(df):
    # scale-free comparison of the two torques
    l = df["left-weight"] * df["left-distance"]
    r = df["right-weight"] * df["right-distance"]
    return np.log(l.replace(0, np.nan)) - np.log(r.replace(0, np.nan))


def torques(df):
    # each side's torque separately
    return pd.DataFrame({"left": df["left-weight"] * df["left-distance"],
                         "right": df["right-weight"] * df["right-distance"]})


PROPOSALS = [("torque_diff", torque_diff), ("torque_sign", torque_sign), ("log_torque_ratio", log_torque_ratio),
             ("torques", torques)]
