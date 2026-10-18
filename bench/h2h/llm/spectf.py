"""Domain features for spectf, written from the schema and description only."""
import numpy as np
import pandas as pd

# OUTSIDE KNOWLEDGE: SPECTF columns F{k}R / F{k}S are perfusion counts for region-of-interest k at Rest and under Stress.
K = range(1, 23)


def _r(df):
    return pd.DataFrame({k: df[f"F{k}R"].astype(float) for k in K})


def _s(df):
    return pd.DataFrame({k: df[f"F{k}S"].astype(float) for k in K})


def stress_rest_diff(df):
    # reversible ischemia: uptake drops under stress relative to rest; summarise per-ROI stress-rest differences
    d = _s(df) - _r(df)
    return pd.DataFrame({"diff_mean": d.mean(axis=1), "diff_min": d.min(axis=1), "diff_std": d.std(axis=1),
                         "n_drop5": (d <= -5).sum(axis=1).astype(float)})


def defect_extent(df):
    # perfusion defects = ROIs with markedly low counts; count of low ROIs at stress and rest (fixed vs inducible)
    r, s = _r(df), _s(df)
    return pd.DataFrame({"n_low_stress": (s < 60).sum(axis=1).astype(float), "n_low_rest": (r < 60).sum(axis=1).astype(float),
                         "min_stress": s.min(axis=1), "min_rest": r.min(axis=1)})


def normalised_uptake(df):
    # counts are relative to the patient's max: deficit of each ROI vs that image's maximum
    r, s = _r(df), _s(df)
    sn = s.div(s.max(axis=1).replace(0, np.nan), axis=0)
    rn = r.div(r.max(axis=1).replace(0, np.nan), axis=0)
    return pd.DataFrame({"stress_norm_mean": sn.mean(axis=1), "rest_norm_mean": rn.mean(axis=1),
                         "stress_norm_min": sn.min(axis=1), "norm_diff_mean": (sn - rn).mean(axis=1)})


def heterogeneity(df):
    # uneven tracer distribution across myocardium suggests regional disease
    r, s = _r(df), _s(df)
    return pd.DataFrame({"stress_cv": s.std(axis=1) / s.mean(axis=1).replace(0, np.nan),
                         "rest_cv": r.std(axis=1) / r.mean(axis=1).replace(0, np.nan)})


def global_means(df):
    # global perfusion at rest and stress and their ratio
    r, s = _r(df), _s(df)
    return pd.DataFrame({"rest_mean": r.mean(axis=1), "stress_mean": s.mean(axis=1),
                         "stress_over_rest": s.mean(axis=1) / r.mean(axis=1).replace(0, np.nan)})


def regional_blocks(df):
    # contiguous ROIs approximate coronary territories; mean stress-rest change in thirds of the ROI list
    d = _s(df) - _r(df)
    return pd.DataFrame({"diff_roi1_7": d[list(range(1, 8))].mean(axis=1), "diff_roi8_15": d[list(range(8, 16))].mean(axis=1),
                         "diff_roi16_22": d[list(range(16, 23))].mean(axis=1)})


PROPOSALS = [("stress_rest_diff", stress_rest_diff), ("defect_extent", defect_extent),
             ("normalised_uptake", normalised_uptake), ("heterogeneity", heterogeneity), ("global_means", global_means),
             ("regional_blocks", regional_blocks)]
