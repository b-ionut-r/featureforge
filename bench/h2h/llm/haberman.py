"""Domain features for haberman, written from the schema and description only."""
import numpy as np
import pandas as pd

A = "Age_of_patient_at_time_of_operation"
Y = "Patients_year_of_operation"
N = "Number_of_positive_axillary_nodes_detected"


def _f(df, c):
    return pd.to_numeric(df[c].astype(object), errors="coerce").astype(float)


def node_burden(df):
    # positive axillary nodes are the dominant breast-cancer prognostic factor; log scale and pN-like bins
    # OUTSIDE KNOWLEDGE: AJCC nodal staging bins 0 / 1-3 / 4-9 / 10+
    n = _f(df, N)
    return pd.DataFrame({"log_nodes": np.log1p(n), "nodes_any": (n > 0).astype(float),
                         "node_stage": np.select([n == 0, n <= 3, n <= 9], [0.0, 1.0, 2.0], 3.0)})


def age_at_op(df):
    # age at diagnosis: very young and elderly patients have worse outcomes; birth year from op year - age
    a = _f(df, A)
    return pd.DataFrame({"age_sq_dev": (a - 52.0) ** 2 / 100.0, "birth_year": _f(df, Y) - a, "age_over_65": (a >= 65).astype(float)})


def era(df):
    # surgical/adjuvant-care era within 1958-1970: later operations had better supportive care
    y = _f(df, Y)
    return pd.DataFrame({"year": y, "late_era": (y >= 65).astype(float)})


def age_x_nodes(df):
    # node burden matters more for older patients with less physiological reserve
    return _f(df, A) * np.log1p(_f(df, N)) / 10.0


def nodes_per_age(df):
    # aggressive tumour in a young patient: nodes relative to age
    return _f(df, N) / _f(df, A).replace(0, np.nan) * 10.0


PROPOSALS = [("node_burden", node_burden), ("age_at_op", age_at_op), ("era", era), ("age_x_nodes", age_x_nodes),
             ("nodes_per_age", nodes_per_age)]
