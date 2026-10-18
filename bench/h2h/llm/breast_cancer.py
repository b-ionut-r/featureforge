"""Domain features for breast_cancer, written from the schema and description only."""
import numpy as np
import pandas as pd


def _mid(s):
    # "40-49" -> 44.5 ; anything unparsable -> NaN
    def f(v):
        try:
            a, b = str(v).split("-")
            return (float(a) + float(b)) / 2.0
        except Exception:
            return np.nan
    return s.astype(object).map(f).astype(float)


def _yes(s):
    return s.astype(object).map({"yes": 1.0, "no": 0.0}).astype(float)


def numeric_ranges(df):
    # binned age / tumour size / involved nodes are ordinal ranges: use their midpoints as numbers
    return pd.DataFrame({"age_mid": _mid(df["age"]), "tumor_mid": _mid(df["tumor-size"]),
                         "nodes_mid": _mid(df["inv-nodes"])})


def stage_score(df):
    # TNM-like burden: tumour size (T) and number of involved nodes (N) together drive recurrence risk
    t = _mid(df["tumor-size"])
    n = _mid(df["inv-nodes"])
    return pd.DataFrame({"t_x_n": t * np.log1p(n), "any_nodes": (n > 1).astype(float),
                         "big_tumor": (t >= 30).astype(float)})


def malignancy_x_spread(df):
    # histological grade combined with nodal spread: high-grade tumours with node involvement recur most
    g = pd.to_numeric(df["deg-malig"].astype(object), errors="coerce").astype(float)
    n = _mid(df["inv-nodes"])
    caps = _yes(df["node-caps"])
    return pd.DataFrame({"grade_x_nodes": g * np.log1p(n), "grade_x_caps": g * caps.fillna(0.5),
                         "grade3": (g == 3).astype(float)})


def aggressive_flags(df):
    # count of adverse prognostic signs: grade 3, nodes involved, capsular invasion, large tumour, irradiation given
    g = pd.to_numeric(df["deg-malig"].astype(object), errors="coerce").astype(float)
    s = ((g == 3).astype(float) + (_mid(df["inv-nodes"]) > 1).astype(float) + _yes(df["node-caps"]).fillna(0)
         + (_mid(df["tumor-size"]) >= 30).astype(float) + _yes(df["irradiat"]).fillna(0))
    return s


def menopause_age(df):
    # premenopausal (younger) patients tend to have more aggressive disease; ordinal menopause status
    m = df["menopause"].astype(object).map({"premeno": 0.0, "lt40": 1.0, "ge40": 2.0}).astype(float)
    return pd.DataFrame({"menopause_ord": m, "premeno_young": ((m == 0) & (_mid(df["age"]) < 45)).astype(float)})


def quadrant(df):
    # tumour location: upper vs lower and central; central/inner tumours drain to different node basins
    q = df["breast-quad"].astype(object).astype(str)
    return pd.DataFrame({"quad_upper": q.str.contains("up").astype(float),
                         "quad_central": (q == "central").astype(float),
                         "quad_same_side": [float(isinstance(b, str) and b in str(qq)) for b, qq in
                                            zip(df["breast"].astype(object), q)]})


PROPOSALS = [("numeric_ranges", numeric_ranges), ("stage_score", stage_score),
             ("malignancy_x_spread", malignancy_x_spread), ("aggressive_flags", aggressive_flags),
             ("menopause_age", menopause_age), ("quadrant", quadrant)]
