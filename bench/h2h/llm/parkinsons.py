"""Domain features for parkinsons, written from the schema and description only."""
import numpy as np
import pandas as pd

# OUTSIDE KNOWLEDGE: V1..V22 mapped to the description's attribute list in the original UCI file order:
# V1 Fo, V2 Fhi, V3 Flo, V4 Jitter(%), V5 Jitter(Abs), V6 RAP, V7 PPQ, V8 DDP, V9 Shimmer, V10 Shimmer(dB),
# V11 APQ3, V12 APQ5, V13 APQ, V14 DDA, V15 NHR, V16 HNR, V17 RPDE, V18 DFA, V19 spread1, V20 spread2, V21 D2, V22 PPE
# (value ranges in the schema agree, e.g. V18 ~0.6-0.8 = DFA, V21 ~2 = D2).


def f0_range(df):
    # pitch range relative to mean pitch: PD voices are monotone (reduced Fhi-Flo span) or show pitch breaks
    fo = df["V1"].replace(0, np.nan)
    return pd.DataFrame({"f0_span_rel": (df["V2"] - df["V3"]) / fo, "fhi_over_fo": df["V2"] / fo,
                         "flo_over_fo": df["V3"] / fo})


def jitter_structure(df):
    # short-term (RAP, 3-period) vs longer-term (PPQ, 5-period) cycle perturbation: tremor shows as longer-term jitter
    return pd.DataFrame({"rap_over_ppq": df["V6"] / df["V7"].replace(0, np.nan),
                         "log_jitter": np.log(df["V4"].clip(lower=1e-6)),
                         "abs_jitter_x_fo": df["V5"] * df["V1"] * 100 / df["V4"].replace(0, np.nan)})


def shimmer_structure(df):
    # amplitude perturbation at 3-, 5- and 11-period windows: slope across windows reflects amplitude tremor
    return pd.DataFrame({"apq3_over_apq11": df["V11"] / df["V13"].replace(0, np.nan),
                         "apq5_over_apq11": df["V12"] / df["V13"].replace(0, np.nan),
                         "log_shimmer": np.log(df["V9"].clip(lower=1e-6))})


def noise_ratio(df):
    # breathiness / incomplete closure: high NHR and low HNR; combine on a log scale
    return pd.DataFrame({"log_nhr": np.log(df["V15"].clip(lower=1e-6)),
                         "hnr_minus_lognhr": df["V16"] + 10 * np.log10(df["V15"].clip(lower=1e-6))})


def dysphonia_composite(df):
    # overall perturbation: jitter, shimmer and noise all rise together in dysphonic voices
    return (np.log(df["V4"].clip(lower=1e-6)) + np.log(df["V9"].clip(lower=1e-6))
            + np.log(df["V15"].clip(lower=1e-6)) - df["V16"] / 10.0)


def nonlinear_combo(df):
    # nonlinear measures (Little et al.): PPE and spread1 capture abnormal pitch variation; RPDE x DFA joint
    return pd.DataFrame({"ppe_x_spread1": df["V22"] * df["V19"], "rpde_x_dfa": df["V17"] * df["V18"],
                         "spread_sum": df["V19"] + 10 * df["V20"]})


def jitter_shimmer_ratio(df):
    # balance of frequency vs amplitude instability
    return df["V4"] / df["V9"].replace(0, np.nan)


PROPOSALS = [("f0_range", f0_range), ("jitter_structure", jitter_structure), ("shimmer_structure", shimmer_structure),
             ("noise_ratio", noise_ratio), ("dysphonia_composite", dysphonia_composite),
             ("nonlinear_combo", nonlinear_combo), ("jitter_shimmer_ratio", jitter_shimmer_ratio)]
