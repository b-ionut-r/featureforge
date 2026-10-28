"""Domain features for t05, written from the schema and description only."""
import numpy as np
import pandas as pd

# OUTSIDE KNOWLEDGE: 908 rows with 6 descriptors matches the QSAR fish-toxicity set: f_1 CIC0 (information index),
# f_2 SM1_Dz(Z) (spectral moment), f_3 GATS1i (Geary autocorrelation, ionization potential), f_4 NdsCH and
# f_5 NdssC (atom-type counts of unsaturated carbons), f_6 MLOGP (Moriguchi logP). Target likely -log(LC50).


def _n(df, c):
    return pd.to_numeric(df[c], errors="coerce")


# Lipophilicity drives baseline narcosis toxicity; nonlinear (bilinear/parabolic) logP response is classic QSAR.
def logp_shape(df):
    lp = _n(df, "f_6")
    return pd.DataFrame({
        "logp_sq": lp ** 2,
        "logp_pos": lp.clip(lower=0),
        "logp_bilinear": lp - np.log10(1 + 10 ** (lp - 5).clip(upper=10)),
    })


# Unsaturated carbon counts: total and presence flags (reactive/aromatic substructures add excess toxicity).
def unsaturation(df):
    a = _n(df, "f_4")
    b = _n(df, "f_5")
    return pd.DataFrame({
        "n_unsat_c": a + b,
        "has_dsch": (a > 0).astype(float),
        "has_dssc": (b > 0).astype(float),
        "dssc_share": b / (a + b).replace(0, np.nan),
    })


# Lipophilicity interactions with size/complexity and polarity descriptors.
def logp_interactions(df):
    lp = _n(df, "f_6")
    cic = _n(df, "f_1")
    sm = _n(df, "f_2")
    gats = _n(df, "f_3")
    return pd.DataFrame({
        "logp_x_cic": lp * cic,
        "logp_x_sm": lp * sm,
        "logp_x_gats": lp * gats,
        "logp_minus_gats": lp - gats,
    })


# Ratios among topological descriptors: heteroatom/polarity content relative to information content.
def descriptor_ratios(df):
    cic = _n(df, "f_1")
    sm = _n(df, "f_2")
    gats = _n(df, "f_3")
    return pd.DataFrame({
        "sm_over_cic": sm / cic.replace(0, np.nan),
        "gats_over_cic": gats / cic.replace(0, np.nan),
        "sm_x_gats": sm * gats,
        "log_sm": np.log1p(sm.clip(lower=0)),
    })


PROPOSALS = [
    ("logp_shape", logp_shape),
    ("unsaturation", unsaturation),
    ("logp_interactions", logp_interactions),
    ("descriptor_ratios", descriptor_ratios),
]
