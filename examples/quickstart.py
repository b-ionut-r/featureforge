"""FeatureForge on a synthetic contest: a main table, a child table and a test file.

    python examples/quickstart.py

Builds features with the whole pipeline, then scores raw columns and FeatureForge's columns with the
same LightGBM on held-out rows.
"""
from pathlib import Path
import tempfile

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from featureforge import build_features

rng = np.random.default_rng(0)
n, n_test = 6000, 2000
main = pd.DataFrame({
    "customer": np.arange(n + n_test),
    "income": rng.gamma(3, 1000, n + n_test),
    "debt": rng.gamma(2, 800, n + n_test),
    "region": rng.choice(list("ABCDEFG"), n + n_test),
})
# A child table: each customer's past payments, some late.
k = rng.integers(0, n + n_test, 50_000)
payments = pd.DataFrame({"customer": k, "month": -rng.integers(1, 36, len(k)),
                         "amount": rng.gamma(2, 100, len(k)), "days_late": rng.poisson(0.3, len(k))})
late = payments.groupby("customer").days_late.mean().reindex(main.customer).fillna(0).to_numpy()
logit = 2.5 * (main.debt / main.income).to_numpy() + 2.0 * late + (main.region == "C") - 2.5
y = (rng.random(n + n_test) < 1 / (1 + np.exp(-logit))).astype(int)

work = Path(tempfile.mkdtemp())
train, test = main.iloc[:n].assign(default=y[:n]), main.iloc[n:]
payments.to_parquet(work / "payments.parquet")
train_path, test_path = build_features(train, test, target="default", id="customer",
                                       tables={"payments": work / "payments.parquet"},
                                       budget=60, out_dir=work / "features")

F_train, F_test = pd.read_parquet(train_path), pd.read_parquet(test_path)
y_test = y[n:]


def score(cols):
    def prep(X):
        X = X[cols].copy()
        for c in X.columns:
            if not pd.api.types.is_numeric_dtype(X[c]):
                X[c] = X[c].astype("category")
        return X
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, verbose=-1).fit(prep(F_train), F_train["default"])
    return roc_auc_score(y_test, m.predict_proba(prep(F_test))[:, 1])


raw = ["income", "debt", "region"]
new = [c for c in F_test.columns if c != "customer"]
print(f"raw columns: AUC {score(raw):.4f} ({len(raw)} columns)")
print(f"FeatureForge: AUC {score(new):.4f} ({len(new)} columns)")
