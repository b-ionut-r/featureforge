"""Domain features for mice_protein, written from the schema and description only."""
import numpy as np
import pandas as pd

PHOSPHO_PAIRS = [
    ("pAKT_N", "AKT_N"), ("pBRAF_N", "BRAF_N"), ("pCAMKII_N", "CAMKII_N"), ("pCREB_N", "CREB_N"),
    ("pELK_N", "ELK_N"), ("pERK_N", "ERK_N"), ("pJNK_N", "JNK_N"), ("pMEK_N", "MEK_N"),
    ("pNR1_N", "NR1_N"), ("pNR2A_N", "NR2A_N"), ("pNR2B_N", "NR2B_N"), ("pRSK_N", "RSK_N"),
    ("pMTOR_N", "MTOR_N"), ("pGSK3B_N", "GSK3B_N"), ("pP70S6_N", "P70S6_N"), ("pNUMB_N", "NUMB_N"),
    ("pS6_N", "S6_N"), ("pPKCAB_N", "PKCA_N"),
]


def _num(df):
    return df.select_dtypes(include=[np.number]).astype(float)


# Phosphorylated / total protein ratios: activation state of each signalling protein, independent of its abundance.
def phospho_ratios(df):
    out = {}
    for p, t in PHOSPHO_PAIRS:
        if p in df.columns and t in df.columns:
            out["ratio_" + p] = np.log(df[p].clip(lower=1e-6) / df[t].replace(0, np.nan).clip(lower=1e-6))
    return pd.DataFrame(out, index=df.index)


# Expression of proteins encoded on the trisomic segment: gene dosage should raise them in trisomic mice.
# OUTSIDE KNOWLEDGE: DYRK1A, ITSN1, SOD1, APP and DSCR1 (RCAN1) map to the region triplicated in the Ts65Dn model.
def trisomic_dosage(df):
    cols = [c for c in ["DYRK1A_N", "ITSN1_N", "SOD1_N", "APP_N", "DSCR1_N"] if c in df.columns]
    x = np.log(df[cols].astype(float).clip(lower=1e-6))
    z = (x - x.mean()) / x.std().replace(0, np.nan)
    return pd.DataFrame({"trisomic_mean_z": z.mean(axis=1), "trisomic_min_z": z.min(axis=1)}, index=df.index)


# MAPK cascade activity (BRAF -> MEK -> ERK -> RSK/ELK): learning activates this pathway.
def mapk_pathway(df):
    cols = ["pBRAF_N", "pMEK_N", "pERK_N", "pRSK_N", "pELK_N"]
    x = np.log(df[cols].astype(float).clip(lower=1e-6))
    z = (x - x.mean()) / x.std().replace(0, np.nan)
    return z.mean(axis=1)


# mTOR / translation pathway activity (AKT -> mTOR -> P70S6 -> S6), affected by memantine and learning.
def mtor_pathway(df):
    cols = ["pAKT_N", "pMTOR_N", "pP70S6_N", "pS6_N", "RAPTOR_N"]
    x = np.log(df[cols].astype(float).clip(lower=1e-6))
    z = (x - x.mean()) / x.std().replace(0, np.nan)
    return z.mean(axis=1)


# NMDA receptor subunits and phosphorylation: memantine is an NMDA antagonist.
# OUTSIDE KNOWLEDGE: memantine acts on NMDA receptors (NR1/NR2A/NR2B subunits).
def nmda_profile(df):
    tot = df[["NR1_N", "NR2A_N", "NR2B_N"]].astype(float)
    ph = df[["pNR1_N", "pNR2A_N", "pNR2B_N"]].astype(float)
    return pd.DataFrame({
        "nmda_total": tot.sum(axis=1),
        "nmda_phospho_frac": ph.sum(axis=1) / tot.sum(axis=1).replace(0, np.nan),
        "nr2a_over_nr2b": df["NR2A_N"] / df["NR2B_N"].replace(0, np.nan),
    }, index=df.index)


# Immediate-early / activity markers (ARC, EGR1, pCFOS, BDNF): stimulation to learn induces them.
def activity_markers(df):
    cols = ["ARC_N", "EGR1_N", "pCFOS_N", "BDNF_N", "pCREB_N"]
    x = np.log(df[cols].astype(float).clip(lower=1e-6))
    z = (x - x.mean()) / x.std().replace(0, np.nan)
    return z.mean(axis=1, skipna=True)


# Sample-level loading: overall log expression level and spread (a per-sample normalization factor).
def sample_level(df):
    x = np.log(_num(df).clip(lower=1e-6))
    return pd.DataFrame({
        "log_expr_mean": x.mean(axis=1), "log_expr_std": x.std(axis=1),
        "n_missing": _num(df).isna().sum(axis=1).astype(float),
    }, index=df.index)


# Apoptosis balance (BAX vs BCL2, BAD, cleaved caspase-9): pro- vs anti-apoptotic signalling.
def apoptosis_balance(df):
    return pd.DataFrame({
        "bax_over_bcl2": df["BAX_N"] / df["BCL2_N"].replace(0, np.nan),
        "pcasp9_over_bcl2": df["pCASP9_N"] / df["BCL2_N"].replace(0, np.nan),
    }, index=df.index)


PROPOSALS = [
    ("phospho_ratios", phospho_ratios),
    ("trisomic_dosage", trisomic_dosage),
    ("mapk_pathway", mapk_pathway),
    ("mtor_pathway", mtor_pathway),
    ("nmda_profile", nmda_profile),
    ("activity_markers", activity_markers),
    ("sample_level", sample_level),
    ("apoptosis_balance", apoptosis_balance),
]
