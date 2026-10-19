"""Domain features for puma32H, written from the schema and description only.

Puma 560 arm dynamics: angular positions (theta), velocities (thetad), torques (tau); the predicted link is not
named, so features are built per joint. dm/da/db are treated as per-link parameter perturbations.
"""
import numpy as np
import pandas as pd

# OUTSIDE KNOWLEDGE: in the DELVE pumadyn family dm* are link mass changes, da* link length changes, db* viscous friction changes.


def trig_angles(df):
    # dynamics depend on angles through sin/cos (gravity loads, inertia matrix), not raw angles
    out = {}
    for i in range(1, 7):
        out[f"sin_t{i}"] = np.sin(df[f"theta{i}"])
        out[f"cos_t{i}"] = np.cos(df[f"theta{i}"])
    return pd.DataFrame(out)


def gravity_terms(df):
    # OUTSIDE KNOWLEDGE: Puma shoulder/elbow gravity torque goes like cos(theta2) and cos(theta2+theta3)
    return pd.DataFrame({"cos_t2": np.cos(df["theta2"]), "cos_t23": np.cos(df["theta2"] + df["theta3"]),
                         "sin_t23": np.sin(df["theta2"] + df["theta3"])})


def torque_per_mass(df):
    # Newton: acceleration ~ torque / (mass-scaled inertia) for each driven joint
    return pd.DataFrame({f"tau{i}_per_m": df[f"tau{i}"] / df[f"dm{i}"].replace(0, np.nan) for i in range(1, 6)})


def torque_per_inertia(df):
    # rotational inertia ~ m * l^2, so torque / (dm * da^2)
    return pd.DataFrame({f"tau{i}_per_I": df[f"tau{i}"] / (df[f"dm{i}"] * df[f"da{i}"] ** 2).replace(0, np.nan)
                         for i in range(1, 6)})


def friction_torque(df):
    # viscous friction opposes motion: db * thetad, and net torque after friction
    out = {}
    for i in range(1, 6):
        fr = df[f"db{i}"] * df[f"thetad{i}"]
        out[f"fric{i}"] = fr
        out[f"net_tau{i}"] = df[f"tau{i}"] - fr
    return pd.DataFrame(out)


def velocity_products(df):
    # Coriolis / centripetal terms are quadratic in joint velocities
    out = {f"thetad{i}_sq": df[f"thetad{i}"] ** 2 for i in range(1, 7)}
    out["thetad2_x_thetad3"] = df["thetad2"] * df["thetad3"]
    out["thetad1_x_thetad2"] = df["thetad1"] * df["thetad2"]
    out["thetad4_x_thetad5"] = df["thetad4"] * df["thetad5"]
    return pd.DataFrame(out)


def kinetic_energy_proxy(df):
    # total arm kinetic energy proxy: sum m_i * l_i^2 * thetad_i^2 (joints 1-5 have params)
    ke = sum(df[f"dm{i}"] * df[f"da{i}"] ** 2 * df[f"thetad{i}"] ** 2 for i in range(1, 6))
    return ke


PROPOSALS = [("trig_angles", trig_angles), ("gravity_terms", gravity_terms), ("torque_per_mass", torque_per_mass),
             ("torque_per_inertia", torque_per_inertia), ("friction_torque", friction_torque),
             ("velocity_products", velocity_products), ("kinetic_energy_proxy", kinetic_energy_proxy)]
