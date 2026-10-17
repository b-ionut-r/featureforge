"""Domain features for airfoil, written from the schema and description only."""
import numpy as np
import pandas as pd

# OUTSIDE KNOWLEDGE: Brooks-Pope-Marcolini airfoil self-noise scaling (Strouhal on displacement thickness,
# Reynolds number, ~U^5 intensity scaling, i.e. 50*log10(U)); air kinematic viscosity ~1.5e-5 m^2/s, c ~ 340 m/s.
NU = 1.5e-5
C_SOUND = 340.0


def strouhal_delta(df):
    # Strouhal number on displacement thickness f*delta/U: sets where the turbulent BL noise spectrum peaks
    return df["frequency"] * df["displacement_thickness"] / df["free_stream_velocity"].replace(0, np.nan)


def log_strouhal(df):
    # BPM spectra are shape functions of log10(St / St_peak)
    st = strouhal_delta(df)
    return np.log10(st.clip(lower=1e-12))


def strouhal_chord(df):
    # chord-based Strouhal f*c/U
    return df["frequency"] * df["chord_length"] / df["free_stream_velocity"].replace(0, np.nan)


def reynolds(df):
    # chord Reynolds number (log): governs laminar vs turbulent boundary layer and tonal noise
    return np.log10((df["free_stream_velocity"] * df["chord_length"] / NU).clip(lower=1e-12))


def velocity_scaling(df):
    # far-field noise intensity ~ U^5 * delta: 50*log10(M) + 10*log10(delta) in dB
    m = df["free_stream_velocity"] / C_SOUND
    return pd.DataFrame({"db_u5": 50 * np.log10(m.clip(lower=1e-12)),
                         "db_delta": 10 * np.log10(df["displacement_thickness"].clip(lower=1e-12)),
                         "db_scale_total": 50 * np.log10(m.clip(lower=1e-12)) + 10 * np.log10(df["displacement_thickness"].clip(lower=1e-12))})


def log_frequency(df):
    # frequency in log scale (octave bands)
    return np.log10(df["frequency"].clip(lower=1e-6))


def thickness_over_chord(df):
    # displacement thickness relative to chord: boundary-layer growth / separation tendency
    return df["displacement_thickness"] / df["chord_length"].replace(0, np.nan)


def aoa_terms(df):
    # angle of attack drives suction-side thickening and stall noise; interaction with Strouhal
    a = df["angle_of_attack"]
    return pd.DataFrame({"aoa_sq": a ** 2, "aoa_x_logst": a * log_strouhal(df), "stalled": (a > 12.5).astype(float)})


PROPOSALS = [("strouhal_delta", strouhal_delta), ("log_strouhal", log_strouhal), ("strouhal_chord", strouhal_chord),
             ("reynolds", reynolds), ("velocity_scaling", velocity_scaling), ("log_frequency", log_frequency),
             ("thickness_over_chord", thickness_over_chord), ("aoa_terms", aoa_terms)]
