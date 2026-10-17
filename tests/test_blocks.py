import numpy as np
import pandas as pd

from featureforge.forge import FeatureForge


def _measures(n, seed, k=10):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({f"m{i}": rng.uniform(1, 10, n) for i in range(k)}), rng


def test_small_table_cv_gate_keeps_a_pair_signal():
    # 800 rows: the cross-validated gate judges; the class depends on two differences of measures.
    X, rng = _measures(800, 0)
    y = ((X.m0 - X.m1) + (X.m2 - X.m3) + rng.normal(scale=0.5, size=len(X)) > 0).astype(int)
    f = FeatureForge(task="binary", time_budget=120, n_jobs=1, verbose=False).fit(X, y)
    assert f.gate_passed_ and getattr(f, "cv_gate_z_", 0) >= 1.0
    assert f.new_columns_
    A, B = f.transform_train(X), f.transform(X)
    assert list(A.columns) == list(B.columns) == list(X.columns) + f.new_columns_


def test_small_table_noise_target_keeps_raw_columns():
    X, rng = _measures(600, 1)
    y = rng.integers(0, 2, len(X))
    f = FeatureForge(task="binary", time_budget=120, n_jobs=1, verbose=False).fit(X, y)
    assert not f.new_columns_
    assert list(f.transform(X).columns) == list(X.columns)
