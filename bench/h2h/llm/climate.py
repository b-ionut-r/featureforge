"""Domain features for climate, written from the schema and description only."""
import numpy as np
import pandas as pd

# All 18 inputs are POP2 ocean-model parameters scaled to [0, 1] by a Latin hypercube design.
VCONST = ["vconst_corr", "vconst_2", "vconst_3", "vconst_4", "vconst_5", "vconst_7"]
VDC = ["bckgrnd_vdc1", "bckgrnd_vdc_ban", "bckgrnd_vdc_eq", "bckgrnd_vdc_psim"]
ALL = VCONST + ["ah_corr", "ah_bolus", "slm_corr", "efficiency_factor", "tidal_mix_max", "vertical_decay_scale",
                "convect_corr"] + VDC + ["Prandtl"]


def viscosity_summary(df):
    # horizontal viscosity parameters jointly set numerical damping; extreme combined values risk CFL-type blow-ups
    X = df[VCONST]
    return pd.DataFrame({"visc_mean": X.mean(axis=1), "visc_max": X.max(axis=1),
                         "vc_corr_x_vc2": df["vconst_corr"] * df["vconst_2"]})


def vertical_diffusion(df):
    # background vertical diffusivity: weak mixing lets instabilities grow; summarise the four background terms
    X = df[VDC]
    return pd.DataFrame({"vdc_mean": X.mean(axis=1), "vdc_min": X.min(axis=1)})


def prandtl_vs_diffusivity(df):
    # Prandtl number = viscosity/diffusivity: high Prandtl with low background diffusivity means large vertical viscosity
    return pd.DataFrame({"prandtl_x_vdc1": df["Prandtl"] * df["bckgrnd_vdc1"],
                         "prandtl_minus_vdc1": df["Prandtl"] - df["bckgrnd_vdc1"]})


def convection_mixing(df):
    # convective adjustment strength vs background mixing: both control how fast static instability is removed
    return pd.DataFrame({"convect_x_vdc1": df["convect_corr"] * df["bckgrnd_vdc1"],
                         "convect_minus_vdc1": df["convect_corr"] - df["bckgrnd_vdc1"]})


def viscosity_vs_mixing(df):
    # high horizontal viscosity coefficient combined with weak vertical mixing (interaction of the two main damping knobs)
    return df["vconst_corr"] * df["vconst_2"] * (1 - df["bckgrnd_vdc1"])


def eddy_params(df):
    # GM eddy diffusivity (ah) vs bolus diffusivity and slope limiter: eddy parameterisation consistency
    return pd.DataFrame({"ah_minus_bolus": df["ah_corr"] - df["ah_bolus"], "ah_x_slm": df["ah_corr"] * df["slm_corr"]})


def tidal_mixing(df):
    # tidal mixing energy times its vertical decay scale: total deep mixing supplied
    return df["tidal_mix_max"] * df["vertical_decay_scale"]


def extremity(df):
    # crashes happen at corners of parameter space: count of parameters near their bounds, distance from centre
    X = df[ALL]
    return pd.DataFrame({"n_extreme": ((X > 0.9) | (X < 0.1)).sum(axis=1).astype(float),
                         "dist_center": np.sqrt(((X - 0.5) ** 2).sum(axis=1))})


PROPOSALS = [("viscosity_summary", viscosity_summary), ("vertical_diffusion", vertical_diffusion),
             ("prandtl_vs_diffusivity", prandtl_vs_diffusivity), ("convection_mixing", convection_mixing),
             ("viscosity_vs_mixing", viscosity_vs_mixing), ("eddy_params", eddy_params),
             ("tidal_mixing", tidal_mixing), ("extremity", extremity)]
