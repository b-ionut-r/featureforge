"""Build one arm's features for one split. Run as a subprocess so a crash or timeout costs one cell.

    python arms.py ARM SPLIT_DIR [--budget S] [--threads N]

SPLIT_DIR holds train.parquet (features + ``__y__``), test.parquet (features only) and meta.json
(task, metric). The arm writes SPLIT_DIR/ARM/{train,test}.parquet and info.json. Every arm's output is
the raw columns plus the columns the tool generated, so the judge sees the same raw input everywhere.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
Y = "__y__"


def _num_new(new_tr, new_te, raw_cols):
    """Keep generated columns that are not raw columns, numeric, finite-ized and not constant."""
    keep = [c for c in new_tr.columns if c not in raw_cols]
    A = new_tr[keep].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    B = new_te[keep].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    ok = [c for c in keep if A[c].nunique(dropna=True) > 1]
    A, B = A[ok].astype("float32"), B[ok].astype("float32")
    names = [f"g{i}" for i in range(len(ok))]
    A.columns, B.columns = names, names
    return A, B, ok


def arm_featureforge(tr, te, meta, budget, threads):
    sys.path.insert(0, os.environ.get("H2H_FF_SRC", "/home/user/ff_main/src"))
    from featureforge import build_features
    out = Path(os.environ["H2H_ARM_DIR"]) / "ff_out"
    prop = os.environ.get("H2H_PROPOSALS")
    kw = {"proposals": prop} if prop else {}
    ptr, pte = build_features(tr, te, Y, task=meta["task"], budget=budget, out_dir=str(out),
                              forge_kw={"n_jobs": threads}, **kw)
    A = pd.read_parquet(ptr).drop(columns=[Y])
    B = pd.read_parquet(pte)[A.columns]
    return A, B, dict(n_new=int(A.shape[1] - tr.shape[1] + 1))


def arm_openfe(tr, te, meta, budget, threads):
    sys.path.insert(0, str(HERE))
    from adapters.openfe_adapter import OpenFEAdapter
    X, y = tr.drop(columns=[Y]), tr[Y]
    X, Xt = X.copy(), te.copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            a, b = (v.astype("string").fillna("__na__").astype(object) for v in (X[c], Xt[c]))
            cats = pd.Index(pd.unique(pd.concat([a, b])))
            X[c], Xt[c] = pd.Categorical(a, categories=cats), pd.Categorical(b, categories=cats)
    ad = OpenFEAdapter("regression" if meta["task"] == "regression" else "classification", budget, 0, n_jobs=threads)
    A = ad.fit_transform(X, y)
    B = ad.transform(Xt)
    A, B, names = _num_new(A, B, set(X.columns))
    return A, B, dict(n_new=len(names), features=[str(f) for f in names[:50]])


def arm_autofeat(tr, te, meta, budget, threads):
    sys.path.insert(0, str(HERE))
    from adapters.autofeat_adapter import AutoFeatAdapter
    X, y = tr.drop(columns=[Y]), tr[Y]
    ad = AutoFeatAdapter("regression" if meta["task"] == "regression" else "classification", budget, 0, n_jobs=threads)
    A = ad.fit_transform(X, y)
    B = ad.transform(te.copy())
    A, B, names = _num_new(A, B, set(X.columns))
    return A, B, dict(n_new=len(names), features=names[:50])


def arm_featuretools(tr, te, meta, budget, threads, max_features=500):
    """Deep Feature Synthesis on one table: each low/mid-cardinality categorical becomes a parent
    table (so DFS builds per-category aggregates of the numeric columns, over train + test rows like
    FeatureForge's label-free statistics), plus arithmetic transforms; then Featuretools' own selection."""
    import featuretools as ft
    from featuretools.selection import (remove_highly_correlated_features, remove_highly_null_features,
                                        remove_single_value_features)
    X = pd.concat([tr.drop(columns=[Y]), te], ignore_index=True)
    X.columns = [f"c{i}" for i in range(X.shape[1])]
    X.insert(0, "_row", np.arange(len(X)))
    cats = [c for c in X.columns[1:] if not pd.api.types.is_numeric_dtype(X[c])]
    for c in cats:
        X[c] = X[c].astype("string").fillna("__na__").astype(object)
    nums = [c for c in X.columns[1:] if c not in cats]
    for c in nums:  # low-cardinality numerics also act as group keys (a categorical copy; the number stays)
        if 2 <= X[c].nunique() <= 32:
            X[f"k_{c}"] = X[c].astype("string").fillna("__na__").astype(object)
            cats.append(f"k_{c}")
    es = ft.EntitySet("h2h").add_dataframe(dataframe_name="t", dataframe=X, index="_row",
                                           logical_types={c: "Categorical" for c in cats})
    parents = [c for c in cats if 2 <= X[c].nunique() <= len(X) // 5][:10]
    for c in parents:
        es = es.normalize_dataframe("t", f"p_{c}", c)
    trans = ["add_numeric", "subtract_numeric", "multiply_numeric", "divide_numeric"] if len(nums) <= 30 else []
    fm, _ = ft.dfs(entityset=es, target_dataframe_name="t", agg_primitives=["mean", "std", "max", "min", "count"],
                   trans_primitives=trans, max_depth=2, n_jobs=1, verbose=False)
    fm = fm.select_dtypes("number").replace([np.inf, -np.inf], np.nan)
    fm = fm.drop(columns=[c for c in X.columns if c in fm.columns])
    fm = remove_highly_null_features(fm)
    fm = remove_single_value_features(fm)
    fm = remove_highly_correlated_features(fm, pct_corr_threshold=0.95)
    if fm.shape[1] > max_features:  # cap by variance rank among the first-depth-first order DFS returns
        fm = fm.iloc[:, :max_features]
    fm = fm.sort_index()
    A, B = fm.iloc[: len(tr)].reset_index(drop=True), fm.iloc[len(tr):].reset_index(drop=True)
    A, B, names = _num_new(A, B, set())
    return A, B, dict(n_new=len(names), features=[str(n) for n in names[:50]])


def llm_features(name, X):
    """Run the proposals written for ``name`` (bench/h2h/llm/<name>.py) on X; returns [(label, frame)]."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"llm_{name}", HERE / "llm" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = []
    for label, fn in mod.PROPOSALS:
        try:
            v = fn(X.copy())
            v = v.to_frame(label) if isinstance(v, pd.Series) else v
            v = v.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
            v.columns = [f"{label}:{c}" for c in v.columns]
            if len(v) == len(X) and v.shape[1]:
                out.append((label, v.reset_index(drop=True)))
        except Exception as e:  # a proposal that fails to run is dropped, as CAAFE does
            print("llm proposal failed", label, repr(e)[:200])
    return out


def _cv_loss(X, y, task, threads, folds=4):
    import lightgbm as lgb
    from sklearn.metrics import log_loss, mean_squared_error
    from sklearn.model_selection import KFold, StratifiedKFold
    X = X.copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("string").fillna("__na__").astype("category")
    X.columns = [f"f{i}" for i in range(X.shape[1])]
    k = int(y.max()) + 1 if task == "multiclass" else 0
    P = dict(objective={"binary": "binary", "multiclass": "multiclass", "regression": "regression"}[task],
             learning_rate=0.08, num_leaves=31, min_child_samples=20, feature_fraction=0.7, bagging_fraction=0.8,
             bagging_freq=1, num_threads=threads, verbose=-1, seed=0, **({"num_class": k} if k else {}))
    sp = (KFold(folds, shuffle=True, random_state=0) if task == "regression"
          else StratifiedKFold(folds, shuffle=True, random_state=0))
    oof = np.zeros((len(X), k)) if k else np.zeros(len(X))
    for a, b in sp.split(X, y if task != "regression" else None):
        m = lgb.train(P, lgb.Dataset(X.iloc[a], y[a]), 2000, valid_sets=[lgb.Dataset(X.iloc[b], y[b])],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[b] = m.predict(X.iloc[b], num_iteration=m.best_iteration)
    if task == "regression":
        return float(mean_squared_error(y, oof) ** 0.5)
    if k:
        return float(log_loss(y, np.clip(oof, 1e-15, 1) / np.clip(oof, 1e-15, 1).sum(1, keepdims=True), labels=np.arange(k)))
    return float(log_loss(y, np.clip(oof, 1e-6, 1 - 1e-6)))


def arm_llm(tr, te, meta, budget, threads):
    """CAAFE-style: domain features an LLM (Claude) wrote from the column names, types, sample rows and the
    dataset description, before any score was seen. Like CAAFE, each proposal is kept only if it lowers the
    cross-validated loss on the training rows, tried in the order proposed."""
    X, y = tr.drop(columns=[Y]), tr[Y].to_numpy()
    allX = pd.concat([X, te], ignore_index=True)
    props = llm_features(meta["name"], allX)
    n = len(X)
    cur = X.reset_index(drop=True)
    best = _cv_loss(cur, y, meta["task"], threads)
    kept, tried = [], []
    for label, v in props:
        cand = pd.concat([cur, v.iloc[:n].reset_index(drop=True)], axis=1)
        loss = _cv_loss(cand, y, meta["task"], threads)
        tried.append((label, loss))
        if loss < best * (1 - 5e-4):
            cur, best = cand, loss
            kept.append(label)
    keep = [c for c in cur.columns if c not in X.columns]
    allnew = pd.concat([v for _, v in props], axis=1) if props else pd.DataFrame(index=range(len(allX)))
    A, B = allnew[keep].iloc[:n].reset_index(drop=True), allnew[keep].iloc[n:].reset_index(drop=True)
    A, B, names = _num_new(A, B, set())
    return A, B, dict(n_new=len(names), kept=kept, tried=tried, n_proposals=len(props))


def arm_ff_snapshot(label):
    """FeatureForge from a frozen copy of a branch's source (FF_SNAP/<label>/src), so a long run keeps
    one version while the branch moves on."""
    def arm(tr, te, meta, budget, threads):
        os.environ["H2H_FF_SRC"] = str(Path(os.environ.get("FF_SNAP", "/home/user/ff_snap")) / label / "src")
        return arm_featureforge(tr, te, meta, budget, threads)
    return arm


def arm_llmff(label):
    """The LLM's kept domain features added to the table first, then FeatureForge (snapshot ``label``) on top:
    does the search still add anything once domain features are in?"""
    def arm(tr, te, meta, budget, threads):
        A, B, info = arm_llm(tr, te, meta, budget, threads)
        A.columns = B.columns = [f"llm{i}" for i in range(A.shape[1])]
        tr2 = pd.concat([tr.reset_index(drop=True), A], axis=1)
        te2 = pd.concat([te.reset_index(drop=True), B], axis=1)
        A2, B2, info2 = arm_ff_snapshot(label)(tr2, te2, meta, budget, threads)
        return A2, B2, dict(info2, n_llm=int(A.shape[1]), kept=info.get("kept"))
    return arm


def arm_ffp(label):
    """FeatureForge (snapshot ``label``) with the LLM's proposals for this table (llm/<name>.py) passed as
    ``proposals``: each kept only if it passes FeatureForge's check on top of the search's output."""
    def arm(tr, te, meta, budget, threads):
        f = HERE / "llm" / f"{meta['name']}.py"
        if f.exists():
            os.environ["H2H_PROPOSALS"] = str(f)
        return arm_ff_snapshot(label)(tr, te, meta, budget, threads)
    return arm


def arm_ffpc(label, proposals=True):
    """As ffp, with the search's output cached per split (H2H_FF_CACHE names the search version, which must
    be the same FeatureForge search code for every snapshot sharing it): proposal-check variants then reuse
    one search run. Same result as ffp when the search code matches (the check runs on the same frames)."""
    def arm(tr, te, meta, budget, threads):
        src = Path(os.environ.get("FF_SNAP", "/home/user/ff_snap")) / label / "src"
        sys.path.insert(0, str(src))
        from featureforge import build_features
        from featureforge import proposals as P
        cache = Path(os.environ["H2H_ARM_DIR"]).parent / f"ffcache_{os.environ.get('H2H_FF_CACHE', 'v10')}"
        if not (cache / "done").exists():
            cache.mkdir(exist_ok=True)
            ptr, pte = build_features(tr, te, Y, task=meta["task"], budget=budget, out_dir=str(cache / "ff_out"),
                                      forge_kw={"n_jobs": threads})
            pd.read_parquet(ptr).to_parquet(cache / "train.parquet")
            pd.read_parquet(pte).to_parquet(cache / "test.parquet")
            (cache / "done").write_text("ok")
        A = pd.read_parquet(cache / "train.parquet")
        y = A.pop(Y).to_numpy()
        B = pd.read_parquet(cache / "test.parquet")[A.columns]
        f = HERE / "llm" / f"{meta['name']}.py"
        kept = []
        if proposals and f.exists():
            frames = P.compute(P.load(f), tr.drop(columns=[Y]), te)
            task = meta["task"]
            yy = (pd.factorize(pd.Series(y).astype("string"), sort=True)[0] if task == "multiclass" else y).astype(float)
            kept = P.gate(A, yy, task, frames, threads=threads)
            cols, n = dict(frames), len(A)
            for lab, names in kept:
                for c in names:
                    A[c] = cols[lab][c].iloc[:n].to_numpy()
                    B[c] = cols[lab][c].iloc[n:].to_numpy()
            print(f"proposals kept: {', '.join(l for l, _ in kept)}", flush=True)
        return A, B, dict(n_new=int(A.shape[1] - tr.shape[1] + 1), kept=[l for l, _ in kept])
    return arm


ARMS = {"llm": arm_llm, "featureforge": arm_featureforge, "openfe": arm_openfe, "autofeat": arm_autofeat,
        "featuretools": arm_featuretools}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm"), ap.add_argument("split_dir")
    ap.add_argument("--budget", type=int, default=900), ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    d = Path(a.split_dir)
    meta = json.loads((d / "meta.json").read_text())
    meta["name"] = (d.parent if d.name == "shuffled" else d).parent.name
    tr, te = pd.read_parquet(d / "train.parquet"), pd.read_parquet(d / "test.parquet")
    out = d / a.arm.replace(":", "_")
    out.mkdir(exist_ok=True)
    os.environ["H2H_ARM_DIR"] = str(out)
    t0 = time.time()
    fn = (arm_ff_snapshot(a.arm.split(":", 1)[1]) if a.arm.startswith("ff:") else
          arm_llmff(a.arm.split(":", 1)[1]) if a.arm.startswith("llmff:") else
          arm_ffp(a.arm.split(":", 1)[1]) if a.arm.startswith("ffp:") else
          arm_ffpc(a.arm.split(":", 1)[1]) if a.arm.startswith("ffpc:") else
          arm_ffpc(a.arm.split(":", 1)[1], proposals=False) if a.arm.startswith("ffc:") else ARMS[a.arm])
    A, B, info = fn(tr, te, meta, a.budget, a.threads)
    info["seconds"] = time.time() - t0
    if a.arm == "featureforge" or a.arm.startswith(("ff:", "llmff:", "ffp:", "ffpc:", "ffc:")):  # FeatureForge returns the full table (it may recode raw columns)
        A.to_parquet(out / "train.parquet"), B.to_parquet(out / "test.parquet")
        info["full"] = True
    else:
        A.to_parquet(out / "train.parquet"), B.to_parquet(out / "test.parquet")
        info["full"] = False
    (out / "info.json").write_text(json.dumps(info, default=str))
    print("ARM_DONE", a.arm, info["seconds"], info.get("n_new"))


if __name__ == "__main__":
    main()
