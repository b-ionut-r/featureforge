"""Domain features for puma8NH, written from the schema and description only."""
import numpy as np
import pandas as pd


def gravity_terms(df):
    # Puma 560 dynamics: gravity torque on links 2/3 goes like cos(theta2) and cos(theta2+theta3)
    return pd.DataFrame({"cos_t2": np.cos(df["theta2"]), "cos_t23": np.cos(df["theta2"] + df["theta3"]),
                         "sin_t23": np.sin(df["theta2"] + df["theta3"])})


def inertia_coupling(df):
    # link-2/3 inertia coupling term varies as cos(theta3); scales how torque turns into acceleration
    return np.cos(df["theta3"])


def centrifugal(df):
    # centrifugal / Coriolis terms are products of joint velocities
    return pd.DataFrame({"td1sq": df["thetad1"] ** 2, "td2sq": df["thetad2"] ** 2,
                         "td2_td3": df["thetad2"] * df["thetad3"],
                         "td23sq_sin3": (df["thetad2"] + df["thetad3"]) ** 2 * np.sin(df["theta3"])})


def torque_net(df):
    # net drive: torque difference and torque relative to gravity load
    return pd.DataFrame({"tau_diff": df["tau1"] - df["tau2"],
                         "tau2_minus_grav": df["tau2"] - np.cos(df["theta2"] + df["theta3"])})


def velocity_norm(df):
    # overall kinetic activity of the arm
    return np.sqrt(df["thetad1"] ** 2 + df["thetad2"] ** 2 + df["thetad3"] ** 2)


PROPOSALS = [("gravity_terms", gravity_terms), ("centrifugal", centrifugal),
             ("inertia_coupling", inertia_coupling), ("torque_net", torque_net), ("velocity_norm", velocity_norm)]
