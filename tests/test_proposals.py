import numpy as np
import pandas as pd

from featureforge import proposals as P


def _data(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.uniform(1, 10, n), "b": rng.uniform(1, 10, n), "c": rng.normal(size=n)})
    y = (X.a / X.b + rng.normal(scale=0.05, size=n) > 1).astype(int).to_numpy()
    return X, y, rng


def test_helpful_proposal_kept_noise_and_broken_dropped():
    X, y, rng = _data()
    tr, te = X.iloc[:1200], X.iloc[1200:]
    props = [("ratio", lambda d: d.a / d.b),
             ("noise", lambda d: pd.Series(np.random.default_rng(5).normal(size=len(d)))),
             ("broken", lambda d: d["missing_col"])]
    frames = P.compute(props, tr, te)
    assert [l for l, _ in frames] == ["ratio", "noise"]
    assert all(len(v) == len(X) for _, v in frames)
    kept = P.gate(tr.reset_index(drop=True), y[:1200], "binary", frames, threads=1)
    assert [l for l, _ in kept] == ["ratio"]
    assert kept[0][1] == ["prop__ratio__ratio"]
