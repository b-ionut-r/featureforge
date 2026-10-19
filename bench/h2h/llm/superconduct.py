"""Domain features for superconduct, written from the schema and description only.

Columns are composition-aggregated elemental properties; features contrast weighted vs unweighted
statistics and build physically motivated combinations.
"""
import numpy as np
import pandas as pd

PROPS = ["atomic_mass", "fie", "atomic_radius", "Density", "ElectronAffinity", "FusionHeat",
         "ThermalConductivity", "Valence"]


def wtd_vs_unwtd(df):
    # stoichiometry-weighted mean over plain mean: whether the dominant elements are high or low in each property
    return pd.DataFrame({f"wtdratio_{p}": df[f"wtd_mean_{p}"] / df[f"mean_{p}"].replace(0, np.nan) for p in PROPS})


def gmean_over_mean(df):
    # geometric/arithmetic mean ratio (<=1): spread/heterogeneity of elements, scale-free
    return pd.DataFrame({f"gm_ratio_{p}": df[f"wtd_gmean_{p}"] / df[f"wtd_mean_{p}"].replace(0, np.nan) for p in PROPS})


def entropy_gap(df):
    # unweighted minus weighted entropy: how unequal the stoichiometric fractions are
    return pd.DataFrame({f"entgap_{p}": df[f"entropy_{p}"] - df[f"wtd_entropy_{p}"] for p in PROPS[:4]})


def isotope_mass(df):
    # OUTSIDE KNOWLEDGE: BCS isotope effect, Tc ~ M^-1/2 (phonon frequency scales with 1/sqrt(mass))
    return 1.0 / np.sqrt(df["wtd_mean_atomic_mass"].clip(lower=1e-6))


def debye_proxy(df):
    # OUTSIDE KNOWLEDGE: Debye-temperature-like stiffness proxy ~ sqrt(melting/fusion energy / (mass * radius^2))
    return np.sqrt(df["wtd_mean_FusionHeat"].clip(lower=0) /
                   (df["wtd_mean_atomic_mass"] * (df["wtd_mean_atomic_radius"] / 100.0) ** 2).replace(0, np.nan))


def valence_electron_count(df):
    # valence electron concentration and its spread: carrier density drives pairing (Matthias rules)
    return pd.DataFrame({"vec": df["wtd_mean_Valence"], "vec_x_nel": df["wtd_mean_Valence"] * df["number_of_elements"],
                         "vec_std_rel": df["wtd_std_Valence"] / df["wtd_mean_Valence"].replace(0, np.nan)})


def atomic_volume(df):
    # mass/density ~ molar volume; vs radius^3 gives packing efficiency (lattice structure)
    vol = df["wtd_mean_atomic_mass"] / df["wtd_mean_Density"].replace(0, np.nan)
    return pd.DataFrame({"molar_vol": vol,
                         "packing": (df["wtd_mean_atomic_radius"] ** 3) / vol.replace(0, np.nan)})


def ionic_contrast(df):
    # ionization energy vs electron affinity spread: ionic/covalent character (cuprate-like oxides have big contrasts)
    return pd.DataFrame({"fie_minus_ea": df["wtd_mean_fie"] - df["wtd_mean_ElectronAffinity"],
                         "range_fie_x_ea": df["range_fie"] * df["range_ElectronAffinity"] / 1e4})


def thermal_cond_log(df):
    # thermal conductivity spans orders of magnitude; log of weighted gmean and range
    return pd.DataFrame({"log_wgm_tc": np.log1p(df["wtd_gmean_ThermalConductivity"].clip(lower=0)),
                         "log_range_tc": np.log1p(df["range_ThermalConductivity"].clip(lower=0))})


PROPOSALS = [("wtd_vs_unwtd", wtd_vs_unwtd), ("gmean_over_mean", gmean_over_mean), ("entropy_gap", entropy_gap),
             ("isotope_mass", isotope_mass), ("debye_proxy", debye_proxy),
             ("valence_electron_count", valence_electron_count), ("atomic_volume", atomic_volume),
             ("ionic_contrast", ionic_contrast), ("thermal_cond_log", thermal_cond_log)]
