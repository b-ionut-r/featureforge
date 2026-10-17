"""Domain features for ailerons, written from the schema and description only."""
import numpy as np
import pandas as pd

SE = [f"SeTime{i}" for i in range(1, 15)]
DSE = [f"diffSeTime{i}" for i in range(1, 15)]


def se_history_stats(df):
    # SeTime1..14 are a history of the Se signal: level, spread and trend summarise it
    X = df[SE].astype(float)
    return pd.DataFrame({"se_mean": X.mean(axis=1), "se_std": X.std(axis=1),
                         "se_trend": df["SeTime14"] - df["SeTime1"], "se_now_minus_mean": df["Se"] - X.mean(axis=1)})


def dse_stats(df):
    # aggregate of the diffs: net change and activity of the Se signal
    X = df[DSE].astype(float)
    return pd.DataFrame({"dse_sum": X.sum(axis=1), "dse_abs": X.abs().sum(axis=1)})


def roll_dynamics(df):
    # aileron command is the roll controller: roll angle, roll rate p and its change
    return pd.DataFrame({"roll_x_p": df["curRoll"] * df["p"], "p_plus_roll": df["p"] + 0.5 * df["curRoll"],
                         "p_plus_droll": df["p"] + 10 * df["diffRollRate"]})


def abs_roll_rel(df):
    # current roll relative to absolute roll reference
    return pd.DataFrame({"roll_minus_abs": df["curRoll"] - df["absRoll"],
                         "roll_over_abs": df["curRoll"] / df["absRoll"].replace(0, np.nan)})


def pitch_alpha(df):
    # longitudinal state: pitch vs angle of attack (flight path), pitch rate times alpha
    return pd.DataFrame({"pitch_minus_alpha": df["curPitch"] - df["alpha"], "q_x_alpha": df["q"] * df["alpha"]})


def climb_terms(df):
    # climb rate and its derivatives combined into a predicted next climb rate
    return pd.DataFrame({"climb_next": df["climbRate"] + df["diffClb"] + 0.5 * df["diffDiffClb"],
                         "sgz_x_climb": df["Sgz"] * np.sign(df["climbRate"])})


def se_alpha(df):
    # Se tracks alpha closely; their ratio isolates the control-surface-specific part
    return df["Se"] / df["alpha"].replace(0, np.nan)


PROPOSALS = [("se_history_stats", se_history_stats), ("dse_stats", dse_stats), ("roll_dynamics", roll_dynamics),
             ("abs_roll_rel", abs_roll_rel), ("pitch_alpha", pitch_alpha), ("climb_terms", climb_terms), ("se_alpha", se_alpha)]
