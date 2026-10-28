"""Domain features for data, written from the schema and description only."""
import numpy as np
import pandas as pd

# OUTSIDE KNOWLEDGE: nine continuous columns where f_3 ~= f_2/f_1 and f_8 is an integer penalty suggest the
# protein tertiary-structure (CASP) physicochemical table: f_1 total surface area, f_2 non-polar exposed area,
# f_3 fractional non-polar exposed area, f_4 fractional area of exposed non-polar part of residue,
# f_5 molecular-mass-weighted exposed area, f_6 average deviation from standard exposed area,
# f_7 Euclidean distance, f_8 secondary-structure penalty, f_9 spatial distribution constraint.
# Target would be RMSD of the decoy; features normalise by protein size.


def size_normalised(df):
    # most raw quantities scale with protein size (total area); per-area versions reflect quality
    a = df["f_1"].replace(0, np.nan)
    return pd.DataFrame({
        "f4_per_area": df["f_4"] / a,
        "f6_per_area": df["f_6"] / a,
        "f7_per_area": df["f_7"] / a,
        "f8_per_area": df["f_8"] / a * 1000.0,
        "mass_per_area": df["f_5"] / a,
    })


def nonpolar_consistency(df):
    # f_3 should equal f_2/f_1 for a consistent structure; residual and polar exposed area
    frac = df["f_2"] / df["f_1"].replace(0, np.nan)
    return pd.DataFrame({
        "np_frac_calc": frac,
        "np_frac_resid": df["f_3"] - frac,
        "polar_area": df["f_1"] - df["f_2"],
    })


def compactness(df):
    # a well-folded protein has area ~ mass^(2/3); excess area or distance indicates unfolded / poor decoy
    # OUTSIDE KNOWLEDGE: surface area of globular proteins scales with M^(2/3) (and f_7 with radius ~ M^(1/3))
    mass_proxy = (df["f_5"] / df["f_1"].replace(0, np.nan)) * df["f_1"]
    return pd.DataFrame({
        "area_over_size23": df["f_1"] / (df["f_5"].clip(lower=1) ** (2.0 / 3.0)),
        "dist_over_sqrt_area": df["f_7"] / np.sqrt(df["f_1"].clip(lower=1)),
        "dist_over_cuberoot_mass": df["f_7"] / (mass_proxy.clip(lower=1) ** (1.0 / 3.0)),
    })


def exposed_deviation(df):
    # deviation from standard exposed area relative to non-polar exposure
    return pd.DataFrame({
        "f6_over_f4": df["f_6"] / df["f_4"].replace(0, np.nan),
        "f6_over_f2": df["f_6"] / df["f_2"].replace(0, np.nan),
        "f4_over_f2": df["f_4"] / df["f_2"].replace(0, np.nan),
    })


def log_scale(df):
    # log of size-like quantities tames heavy right tails
    return pd.DataFrame({
        "log_f1": np.log1p(df["f_1"].clip(lower=0)),
        "log_f5": np.log1p(df["f_5"].clip(lower=0)),
        "log_f8": np.log1p(df["f_8"].clip(lower=0)),
    })


def constraint_interactions(df):
    # spatial-distribution constraint and secondary-structure penalty combined with size
    return pd.DataFrame({
        "f9_x_logf1": df["f_9"] * np.log1p(df["f_1"].clip(lower=0)),
        "f8_over_f7": df["f_8"] / df["f_7"].replace(0, np.nan),
        "f9_x_f3": df["f_9"] * df["f_3"],
    })


PROPOSALS = [
    ("size_normalised", size_normalised),
    ("nonpolar_consistency", nonpolar_consistency),
    ("compactness", compactness),
    ("exposed_deviation", exposed_deviation),
    ("log_scale", log_scale),
    ("constraint_interactions", constraint_interactions),
]
