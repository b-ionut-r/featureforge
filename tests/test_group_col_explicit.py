"""An explicit group column is honoured: never replaced by a time column found in the data, never silently ignored."""
import numpy as np
import pandas as pd
import pytest

from featureforge import FeatureForge


def _data(n_groups=80, per=10, seed=0):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_groups), per)
    skill = rng.normal(size=n_groups)[g]
    X = pd.DataFrame({"pid": g, "week": np.tile(np.arange(per), n_groups).astype(float),
                      "a": rng.normal(size=len(g)), "skill": skill})
    y = skill + 0.5 * X["a"].to_numpy() + rng.normal(scale=0.5, size=len(g))
    return X, y


def test_explicit_group_col_beats_detected_time_col():
    X, y = _data()
    tr = X["pid"] < 60
    U = X[~tr].copy()
    U["week"] += 100.0                      # test rows look later in time, as a later season's rows would
    ff = FeatureForge(time_budget=5, n_rounds=1, group_col="pid", verbose=False)
    ff.fit(X[tr].reset_index(drop=True), y[tr.to_numpy()], X_unlabeled=U.reset_index(drop=True))
    assert ff.group_col_ == "pid"
    assert ff.time_col_ is None


def test_missing_explicit_group_col_raises():
    X, y = _data()
    with pytest.raises(ValueError, match="group_col"):
        FeatureForge(time_budget=5, n_rounds=1, group_col="nfl_id", verbose=False).fit(X, y)
